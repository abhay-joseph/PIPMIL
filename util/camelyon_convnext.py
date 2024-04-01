import os
import os.path
import pathlib
import types

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms

# import utils_augemntation

MODEL_PATH = './tenpercent_resnet18.ckpt'
RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
NUM_CLASSES = 4  # only used if RETURN_PREACTIVATION = False


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


model = torchvision.models.__dict__['convnext_tiny'](pretrained=False)

state = torch.load(MODEL_PATH, map_location='cuda:0')

state_dict = state['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

model = load_model_weights(model, state_dict)


# model.fc = torch.nn.Sequential()

def resnet_forward_impl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


model._forward_impl = types.MethodType(resnet_forward_impl, model)

model = model.cuda()
model.eval()

normalize_to_tensor_transform = transforms.Compose([
    # utils_augemntation.HistoNormalize(),
    transforms.ToTensor(),
])


class ImageDataset(Dataset):
    def __init__(self, dataset_path):
        self.files = list(str(p) for p in pathlib.Path(dataset_path).rglob('*.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = pil_loader(self.files[item])
        img = normalize_to_tensor_transform(img)
        return img

classes = [0,1]
root = '/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/PIPNet/data/CAMELYON/dataset'
data = []

for split in os.listdir(root):
    split_root = os.path.join(root,split)
    for class_name in os.listdir(split_root):
        class_path = os.path.join(split_root, class_name)
        data += [os.path.join(class_path,bag_path) for bag_path in os.listdir(class_path)]

for dataset_path in data:
    out_file = os.path.join(dataset_path, 'embeddings.pth')
    if not os.path.isdir(dataset_path):
        print('no data for', dataset_path)
        continue
    if os.path.exists(out_file):
        print('skipping already processed', dataset_path)
        continue
    print('processing', dataset_path)

    loader = torch.utils.data.DataLoader(ImageDataset(dataset_path),
                                         batch_size=100,
                                         num_workers=8,
                                         pin_memory=True,
                                         prefetch_factor=4)
    embeds = []
    with torch.no_grad():
        for idx, photos in enumerate(loader):
            photos = photos.cuda()
            out = model(photos)
            out = out.cpu().detach()
            embeds.append(out)
            print('batch {} of {}'.format(idx, len(loader.dataset) // 100))

    embeds = torch.cat(embeds, dim=0)
    torch.save(embeds, out_file)
