import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
from torch import Tensor
import torch.utils.checkpoint as checkpoint
from memory_profiler import profile


class PIPMIL(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        # self.quant = torch.ao.quantization.QuantStub()
        # self.dequant = torch.ao.quantization.DeQuantStub()
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    # Only designed for batch size = 1 or uniform sized bags
    # @profile
    def forward(self, xs, max_indices=None, inference=False, vis=False):
        batch_size = xs.size(0)
        patches_per_bag = xs.size(1)
        chunk_size = 50
        stored_pooled_scores = []
        max_pooled_indices = []
        stored_pooled_features = []

        # Forward pass without gradients
        if max_indices==None or inference==True:
            with torch.autocast(device_type="cuda", enabled=True):
                for i in range(batch_size):
                    stored_pooled = []
                    # stored_features = []
                    for j in range(0, patches_per_bag, chunk_size):
                        chunk = xs[i, j:j+chunk_size]
                        # chunk  = self.quant(chunk)
                        features = self._net(chunk)
                        proto_features = self._add_on(features)
                        # proto_features = self.dequant(proto_features)
                        pooled = self._pool(proto_features)
                        # stored_features.append(proto_features)
                        stored_pooled.append(pooled)
                    # stored_features = torch.cat(stored_features, dim=0)
                    stored_pooled = torch.cat(stored_pooled, dim=0)

                    bag_pooled, max_instances =  stored_pooled.max(dim=0)
                    # stored_pooled_features.append(stored_features)
                    stored_pooled_scores.append(bag_pooled)
                    max_pooled_indices.append(max_instances)
                # stored_pooled_features = torch.stack(stored_pooled_features,dim=0)
                stored_pooled_scores = torch.stack(stored_pooled_scores,dim=0)
                max_pooled_indices = torch.stack(max_pooled_indices, dim=0)
                
                clamped_pooled = torch.where(stored_pooled_scores < 0.1, torch.tensor(0., device=stored_pooled_scores.device), stored_pooled_scores)  #during inference, ignore all prototypes that have 0.1 similarity or lower
                out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
                return proto_features, clamped_pooled, out, max_pooled_indices 
            
        # if inference:
        #     clamped_pooled = torch.where(stored_pooled_scores < 0.1, torch.tensor(0., device=stored_pooled_scores.device), stored_pooled_scores)  #during inference, ignore all prototypes that have 0.1 similarity or lower
        #     out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
        #     return stored_features, clamped_pooled, out
        else:
            # Forward pass with gradients for the max-pooled instances
            max_pooled_features = []
            max_bag_pooled_features = []
            max_proto_features = []
            for i in range(batch_size):
                bag_max=xs[i]
                max_indices_i = torch.unique(max_indices)
                xs_max = torch.index_select(bag_max, 0, max_indices_i)
                features_max = self._net(xs_max)
                proto_features_max = self._add_on(features_max)
                max_proto_features.append(proto_features_max)
                pooled_max = self._pool(proto_features_max)
                max_pooled_features.append(pooled_max)

                bag_pooled_max =  pooled_max.max(dim=0)[0]
                max_bag_pooled_features.append(bag_pooled_max)
            max_proto_features = torch.stack(max_proto_features, dim=0)
            max_pooled_features = torch.stack(max_pooled_features, dim=0)
            max_bag_pooled_features = torch.stack(max_bag_pooled_features, dim=0)

            out = self._classification(max_bag_pooled_features) #shape (bs*2, num_classes) 
            
            return max_proto_features, max_pooled_features, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


def get_network_MIL(num_classes: int, args: argparse.Namespace): 
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
        
    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes


    