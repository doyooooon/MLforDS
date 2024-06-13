import numpy as np
from PIL import Image, ImageFile
import os
import json
import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault
from torch.utils.data import Dataset
from collections import Counter
import torch.utils.data as data
import torchvision.transforms.functional as F

# Parameters for data
vet_mean = (0.5, 0.5, 0.5) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
vet_std = (0.5, 0.5, 0.5) # equals np.std(train_set.train_data, axis=(0,1,2))/255

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Augmentations.

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

class GrayTo3Channel(object):
    def __call__(self, img):
        return img.convert('RGB')

# Augmentations.
transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=vet_mean, std=vet_std)
        ])

transform_strong = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=vet_mean, std=vet_std)
        ])
#transform_strong.transforms.insert(0, RandAugment(3, 4))
#transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=vet_mean, std=vet_std)
        ])

class TransformMix:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

# Train dataset - 전체 dataset 구축
class Bone_dataset(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'Raw/TS2_DOG')
        self.json_dir = os.path.join(root_dir, 'Labeled/TL2_DOG')

        self.image_files = os.listdir(self.image_dir)
        self.data = []
        self.targets = []
        self.disease = []

        for image_file in self.image_files:
            image_path = os.path.join(self.image_dir, image_file)
            json_file = image_file.replace('.jpg', '.json')
            json_path = os.path.join(self.json_dir, json_file)

            with open(json_path, 'r') as f:
                json_data = json.load(f)
                disease_name = json_data['metadata']['Disease-Name']
                disease_yes = json_data['metadata']['Disease']
            
            # 질병 정보가 'ABN'인 경우에만 데이터셋에 추가
            if disease_yes == 'ABN':
                self.data.append(Image.open(image_path))
                self.targets.append(int(disease_name[-1])-1)
                self.disease.append(disease_yes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        disease_name = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, disease_name, idx
    
# Validation dataset - 전체 dataset 구축
class Bone_dataset_val(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'Raw/VS2_DOG')
        self.json_dir = os.path.join(root_dir, 'Labeled/VL2_DOG')

        self.image_files = os.listdir(self.image_dir)
        self.data = []
        self.targets = []
        self.disease = []

        for image_file in self.image_files:
            image_path = os.path.join(self.image_dir, image_file)
            json_file = image_file.replace('.jpg', '.json')
            json_path = os.path.join(self.json_dir, json_file)

            with open(json_path, 'r') as f:
                json_data = json.load(f)
                disease_name = json_data['metadata']['Disease-Name']
                disease_yes = json_data['metadata']['Disease']
            
            # 질병 정보가 'ABN'인 경우에만 데이터셋에 추가
            if disease_yes == 'ABN':
                self.data.append(Image.open(image_path))
                self.targets.append(int(disease_name[-1])-1)
                self.disease.append(disease_yes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        disease_name = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, disease_name, idx


def get_vet_original(root, l_samples, u_samples, name,
                transform_train=transform_train,
                transform_strong=transform_strong,
                 transform_val=transform_val):
    print('start')
    dataset = Bone_dataset(root_dir=os.path.join(root, 'Training'),
                            transform=transform_train)
    print('done')
    print(dataset[0][0].shape)
    print('start')
    test_dataset = Bone_dataset_val(root_dir=os.path.join(root, 'Validation'),
                                transform=transform_val)
    print('done')
    print(test_dataset[0][0].shape)
    print(f'유효한 train data 개수: {len(dataset)}')
    print(f'유효한 test data 개수: {len(test_dataset)}')
    
    counter = Counter(dataset.targets)
    counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    train_spilt_order = list(counter.keys())
    for num, count in counter.items():
        print(f"숫자 {num}은(는) {count}번 등장했습니다.")

    train_labeled_idxs, train_unlabeled_idxs = train_split_vet(dataset.targets,
                                                    l_samples,
                                                    u_samples,
                                                    train_spilt_order,
                                                    fix=False)
    
    print(f'Train & Label yes: {len(train_labeled_idxs)}')
    print(f'Train & Label no: {len(train_unlabeled_idxs)}')
    print(f'Total train data num: {len(train_labeled_idxs)+len(train_unlabeled_idxs)}')
    if 'remix' in name:
        train_labeled_dataset = Bone_labeled(root_dir=os.path.join(root, 'Training'),
                                         indexs=train_labeled_idxs, transform=transform_strong)
    else:
        train_labeled_dataset = Bone_labeled(root_dir=os.path.join(root, 'Training'),
                                         indexs=train_labeled_idxs, transform=transform_train)
    if 'remix' in name or 'fix' in name:
        train_unlabeled_dataset = Bone_unlabeled(root_dir=os.path.join(root, 'Training'),
                                                indexs=train_unlabeled_idxs, transform=TransformTwice(transform_train, transform_strong))
    else:
        train_unlabeled_dataset = Bone_unlabeled(root_dir=os.path.join(root, 'Training'),
                                                indexs=train_unlabeled_idxs, transform=TransformMix(transform_train))
    test_dataset = Bone_labeled_val(root_dir=os.path.join(root, 'Validation'),
                                indexs=list(range(len(test_dataset))), transform=transform_val)
    print(f'Total test num: {len(test_dataset)}')
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split_vet(labels, n_labeled_per_class, n_unlabeled_per_class, train_spilt_order, fix=False):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    i = 0
    for class_num in train_spilt_order:
        print(f'now trial: label {class_num}')
        idxs = np.where(labels == class_num)[0]
        print(len(idxs))
        print(len(idxs[:n_labeled_per_class[i]]))
        print(len(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]]))
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        if fix:
            train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        else:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        i += 1
    return train_labeled_idxs, train_unlabeled_idxs

# Train dataset - Labaled 구축
class Bone_labeled(Bone_dataset):

    def __init__(self, root_dir, indexs, transform=None):
        super(Bone_labeled, self).__init__(root_dir=root_dir, transform=transform)

        self.data = [self.data[i] for i in indexs]
        self.targets = [self.targets[i] for i in indexs]

# Train dataset - Unlabeled 구축
class Bone_unlabeled(Bone_dataset):

    def __init__(self, root_dir, indexs, transform=None):
        super(Bone_unlabeled, self).__init__(root_dir=root_dir, transform=transform)

        self.data = [self.data[i] for i in indexs]
        self.targets = [-1 for _ in indexs]

# Validation dataset - Labaled 구축
class Bone_labeled_val(Bone_dataset_val):

    def __init__(self, root_dir, indexs, transform=None):
        super(Bone_labeled_val, self).__init__(root_dir=root_dir, transform=transform)
        
        self.data = [self.data[i] for i in indexs]
        self.targets = [self.targets[i] for i in indexs]
