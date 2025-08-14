from tqdm import tqdm
import os
import copy
import csv
import random

import PIL

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image
from sklearn.model_selection import train_test_split
from transforms_factory import transforms_imagenet_train_aug

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

def all2one_target_transform(label, target=0):
    t = type(label)
    if t == int:
        return target
    elif t == torch.Tensor:
        return torch.ones_like(label, dtype=label.dtype) * target
    elif t == list:
        return [target] * len(label)

def all2all_target_transform(label, n_classes):
    t = type(label)
    if t == list:
        return [(e + 1) % n_classes  for e in label]
    else:
        return (label + 1) % n_classes
        
def get_target_transform(args):
    if args.attack_mode == 'all2all':
        return lambda x: all2all_target_transform(x, args.nb_classes)
    elif args.attack_mode == 'all2one' or args.attack_mode == 'clean_label':
        return lambda x: all2one_target_transform(x, args.attack_label)
    else:
        raise Exception(f'Invalid attack mode {args.attack_mode}')

class GTSRB(torch.utils.data.Dataset):
    def __init__(self, data_root, train, transform, min_width=0):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transform = transform

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: #only load images if more than certain width
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = PIL.Image.open(self.images[index])
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label        

class CelebA_attr(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train, transform):
        if is_train:
            split = "train"
        else:
            split = "test"
            
        self.dataset = torchvision.datasets.CelebA(root=data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transform = transform
        self.split = split
        self.label = [self._convert_attributes(target[self.list_attributes]) for (_, target) in self.dataset]

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        target = self.label[index]
        return (input, target)

class BackdoorDataset(VisionDataset):

    def __init__(self, dataset, label_transform, 
                 portion=0.1, transform=None, target_transform=None,
                 random_seed=99, return_orig=True, selected_idx=None,
                 clean_label=False, train=True, target_cls=None, return_is_poison=False):
        self.dataset = dataset
        self.target_transform = target_transform
        self.label_transform = label_transform
        self.transform = transform
        self.to_pil_image = ToPILImage()
        self.is_train = train
        self.return_is_poison = return_is_poison

        if clean_label and train:
            class_idx = torch.tensor(dataset.targets) == target_cls
            clean_label_idx = torch.arange(len(dataset))[class_idx].numpy()
            if selected_idx is not None:
                assert class_idx[selected_idx].prod().item() == 1
        else:
            clean_label_idx = range(len(dataset))
        
        if selected_idx is None:
            if portion == 0:
                poisoned_indices = []
            elif portion == 1.0:
                poisoned_indices = clean_label_idx
            else:
                poisoned_indices, _ = train_test_split(clean_label_idx, train_size=portion, random_state=random_seed)
                
            self.poisoned_indices = set(poisoned_indices)
        else:
            self.poisoned_indices = set(selected_idx)
        print('Number of poisoned samples: ', len(self.poisoned_indices))
        self.num_bd = len(self.poisoned_indices)
        self.channels, self.width, self.height = self.dataset[0][0].shape
        self.return_orig = return_orig
        
    def __getitem__(self, idx):
        orig_img, orig_target = self.dataset[idx]

        if idx in self.poisoned_indices:
            target = self.label_transform(orig_target)
            img = self.add_trigger(orig_img)
        else:
            target = orig_target
            img = orig_img

        if self.transform is not None:
            orig_img = self.transform(orig_img)
            img = self.transform(img)


        if self.target_transform is not None:
            orig_target = self.target_transform(orig_target)
            target = self.target_transform(target)
        

        if self.return_orig:
            if self.return_is_poison:
                return img, target, orig_img, orig_target, idx in self.poisoned_indices
            else:
                return img, target, orig_img, orig_target
        else:
            return img, target

    def __len__(self):
        return len(self.dataset)
    
    def add_trigger(self, img):
        raise Exception('Trigger adding not implemented')


class PoisonedDataset(VisionDataset):

    def __init__(self, dataset, label_transform, 
                 poisoned_pixel_val=1, portion=0.1, pattern_width=2, transform=None, target_transform=None,
                 loc_w=0, loc_h=1,
                 random_seed=99, return_orig=True, selected_idx=None,
                 clean_label=False, train=True, target_cls=None, trigger_type='easy', return_is_poison=False):
        self.dataset = dataset
        self.target_transform = target_transform
        self.label_transform = label_transform
        self.pattern_width = pattern_width
        self.transform = transform
        self.to_pil_image = ToPILImage()
        self.loc_w = loc_w
        self.loc_h = loc_h
        self.is_train = train
        self.return_is_poison = return_is_poison
        self.trigger_type = trigger_type
        assert trigger_type in ['easy', 'hard']
        self.trigger_mask = [
                ((-1, -1), 1),
                ((-1, -2), -1),
                ((-1, -3), 1),
                ((-2, -1), -1),
                ((-2, -2), 1),
                ((-2, -3), -1),
                ((-3, -1), 1),
                ((-3, -2), -1),
                ((-3, -3), -1)
            ]

        if clean_label and train:
            class_idx = torch.tensor(dataset.targets) == target_cls
            clean_label_idx = torch.arange(len(dataset))[class_idx].numpy()
            if selected_idx is not None:
                assert class_idx[selected_idx].prod().item() == 1
        else:
            clean_label_idx = range(len(dataset))
        
        if selected_idx is None:
            if portion == 0:
                poisoned_indices = []
            elif portion == 1.0:
                poisoned_indices = clean_label_idx
            else:
                poisoned_indices, _ = train_test_split(clean_label_idx, train_size=portion, random_state=random_seed)
                
            self.poisoned_indices = set(poisoned_indices)
        else:
            self.poisoned_indices = set(selected_idx)
        poisoned_indices = np.zeros(len(dataset)).astype(bool)
        poisoned_indices[list(self.poisoned_indices)] = True
        self.poisoned_indices = poisoned_indices
        
        self.num_bd = self.poisoned_indices.sum()#len(self.poisoned_indices)
        print('Number of poisoned samples: ', self.num_bd)
        self.poisoned_pixel_val = poisoned_pixel_val
        
        self.channels, self.width, self.height = 3, 224, 224
        self.return_orig = return_orig

    def get_data(self, idx):
        img, target = self.dataset[idx]
        poisoned_img = self.__add_trigger(img)
        return img, poisoned_img, target

    def get_prepoison_data(self):
        imgs, poisoned_imgs, targets = [], [], []
        
        for img, target in tqdm(self.dataset):
            imgs.append(img)
            poisoned_imgs.append(self.__add_trigger(img))
            targets.append(target)
        return imgs, poisoned_imgs, targets
        
    def __getitem__(self, idx):
        orig_img, orig_target = self.dataset[idx]

        if self.poisoned_indices[idx]:
            target = self.label_transform(orig_target)
            img = self.__add_trigger(orig_img)
        else:
            target = orig_target
            img = orig_img


        if self.transform is not None:
            orig_img = self.transform(orig_img)
            img = self.transform(img)


        if self.target_transform is not None:
            orig_target = self.target_transform(orig_target)
            target = self.target_transform(target)
        
        if self.return_orig:
            if self.return_is_poison:
                return img, target, orig_img, orig_target, idx in self.poisoned_indices
            else:
                return img, target, orig_img, orig_target
        else:
            return img, target

    def __len__(self):
        return len(self.dataset)
    
    def __add_trigger(self, img):
        new_img = copy.deepcopy(img)
        h, w = 5, 5
        if self.trigger_type == 'hard':
            for (x, y), value in self.trigger_mask:
                new_img[:, x-h, y-w] = value
        elif self.trigger_type == 'easy':
            for c in range(self.channels):
                assert self.width-self.loc_w-self.pattern_width >= 0
                for i in range(self.pattern_width):
                    assert self.height-self.loc_h-i >= 0 and self.height-self.loc_h-i < self.height                

                    new_img[c, 
                            self.height-self.loc_h-i-h, 
                            self.width-self.loc_w-self.pattern_width-w:self.width-self.loc_w-w] = self.poisoned_pixel_val 
        return new_img
    
class BlendedPoisonedDataset(BackdoorDataset):

    def __init__(self, dataset, label_transform, portion=0.1, transform=None, target_transform=None,
                  random_seed=99, return_orig=True, selected_idx=None, clean_label=False, train=True, target_cls=None,
                  blended_rate=0.2, input_size=32):
        super().__init__(dataset, label_transform, portion, transform, target_transform, random_seed, return_orig, selected_idx, clean_label, train, target_cls)
        self.blended_rate = blended_rate
        blended_img = Image.open('resources/hello_kitty.jpeg')
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=2),
            transforms.ToTensor()
        ])
        self.blended_img = transform(blended_img)

    def add_trigger(self, img):
        return img * (1 - self.blended_rate) + self.blended_rate * self.blended_img
    
class SIGPoisonedDataset(BackdoorDataset):
    # clean label attack
    def __init__(self, dataset, label_transform, portion=0.1, transform=None, target_transform=None,
                  random_seed=99, return_orig=True, selected_idx=None, clean_label=True, train=True, target_cls=None,
                  delta=20, f=6, input_size=32, return_is_poison=False):
        super().__init__(dataset, label_transform, portion, transform, target_transform, random_seed, return_orig, selected_idx, clean_label, train, target_cls, return_is_poison)
        self.delta = delta
        self.f = f
        self.pattern = torch.zeros((3, input_size, input_size))
        m = self.pattern.shape[1]
        for i in range(int(input_size)):
            for j in range(int(input_size)):
                self.pattern[:, i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m) / 255


    def add_trigger(self, img):
        
        img = (img + self.pattern).clamp(0, 1)
        return img
    
class WaNetPoisonedDataset(VisionDataset):

    def __init__(self, dataset, label_transform, 
                 portion=0.1, transform=None, target_transform=None,
                 k=4, s=0.5, grid_rescale=1, cross_ratio=2, rate_bd=0.1,
                 input_height=32, identity_grid=None, noise_grid=None,
                 random_seed=99, return_orig=True):
        self.dataset = dataset
        self.target_transform = target_transform
        self.label_transform = label_transform
        self.transform = transform
        self.to_pil_image = ToPILImage()
        
        self.k=k
        self.s=s
        self.grid_rescale=grid_rescale
        self.cross_ratio=cross_ratio
        self.rate_bd=rate_bd
        self.input_height = input_height
        assert identity_grid is not None and noise_grid is not None
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
        self.return_orig = return_orig

        self.portion = portion
        if portion == 0:
            poisoned_indices = []
        elif portion == 1.0:
            poisoned_indices = range(len(dataset))
        else:
            poisoned_indices, _ = train_test_split(range(len(dataset)), train_size=portion, random_state=random_seed)
            
        self.poisoned_indices = set(poisoned_indices)
        self.num_bd = len(poisoned_indices)
        self.channels, self.width, self.height = self.dataset[0][0].shape
        import kornia.augmentation as A
        class ProbTransform(torch.nn.Module):
            def __init__(self, f, p=1):
                super(ProbTransform, self).__init__()
                self.f = f
                self.p = p

            def forward(self, x):  # , **kwargs):
                if random.random() < self.p:
                    return self.f(x)
                else:
                    return x
        class PostTensorTransform(torch.nn.Module):
            def __init__(self):
                super(PostTensorTransform, self).__init__()
                self.random_crop = ProbTransform(
                    A.RandomCrop((input_height, input_height), padding=4), p=0.8
                )
                # self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
                self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

            def forward(self, x):
                for module in self.children():
                    x = module(x)
                return x
            
        self.post_transform = PostTensorTransform()
        
    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx][0])
        orig_img, orig_target = self.dataset[idx]

        if idx in self.poisoned_indices:
            target = self.label_transform(orig_target)
            img = self.__add_trigger(orig_img)
        else:
            target = orig_target
            img = orig_img
        img = self.post_transform(img).squeeze(0)

        if self.transform is not None:
                          
            orig_img = self.transform(orig_img)
            img = self.transform(img)

        if self.target_transform is not None:
            orig_target = self.target_transform(orig_target)
            target = self.target_transform(target)
        
        if self.return_orig:
            return img, target, orig_img, orig_target
        else:
            return img, target

    def __len__(self):
        return len(self.dataset)
    
    def __add_trigger(self, img):
        new_img = copy.deepcopy(img)
        bs = 1
        
        num_bd = 1#int(bs * self.rate_bd)
        num_cross = int(num_bd * self.cross_ratio)

        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        inputs_bd = F.grid_sample(new_img.unsqueeze(0), grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True).squeeze(0)
        return inputs_bd

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        inputs, targets = torch.stack(inputs), torch.tensor(targets)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * self.portion)
        num_cross = int(num_bd * self.cross_ratio)
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, self.input_height, self.input_height, 2) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(targets[:num_bd]) * 0 # label 0


        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

        total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
        total_inputs = self.post_transform(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        return total_inputs, total_targets, None, None
    
    def collate_fn_eval(self, batch):
        inputs, targets = zip(*batch)
        inputs, targets = torch.stack(inputs), torch.tensor(targets)
        bs = inputs.shape[0]

        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(bs, self.input_height, self.input_height, 2) * 2 - 1
        grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(targets) * 0
        return inputs_bd, targets_bd, inputs, targets


def random_rotate(x, y):
    angle = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240]) * 1.0
    return 
    

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset, num_classes):
        self.idataset = target_dataset
        self.odataset = outter_dataset
        self.num_classes = num_classes

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            labels = self.num_classes
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)
    
def build_extra_dataset(args, target_cls, is_train=True, transform=None, num_classes=-1):
    dset, _ = get_dataset(args, is_train, prepoison_transform=transform)
    labels = torch.tensor(dset.targets)
    mask = labels == target_cls
    indices = mask.nonzero().squeeze()
    sub_dset = Subset(dset, indices)
    
    tinyimagenet = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'tiny-imagenet-200', 'train' if is_train else 'val'), # test?
            transform=transform)
    if num_classes > 0:
        ood_labels = torch.tensor(tinyimagenet.targets)
        selected_idx = (ood_labels < num_classes).nonzero().squeeze()
        tinyimagenet = Subset(tinyimagenet, selected_idx)
    return concoct_dataset(sub_dset, tinyimagenet, num_classes=num_classes)

def build_aug_transform(args):
    # this should always dispatch to transforms_imagenet_train
    transform = transforms_imagenet_train_aug(
            img_size=args.input_size,
            color_jitter=args.color_jitter,
            interpolation=args.train_interpolation,
            auto_augment=args.aa,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            to_tensor=True,
            normalize=True
        )
    return transform

def build_simple_aug_transform(args, to_tensor=False):
    # this should always dispatch to transforms_imagenet_train
    mean, std = ((CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)) if args.mean is None else (args.mean, args.std)
    if args.attack_type == 'wanet':
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=2, antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = [transforms.Resize(args.input_size, interpolation=2, antialias=True)]
        transform.append(transforms.RandomCrop((args.input_size, args.input_size), padding=5))
        transform.append(transforms.RandomRotation(10))
        if args.data_set != 'GTSRB':
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        if to_tensor:
            transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform)

    return transform

def build_pm1_aug_transform(args):
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if args.mean is None else (args.mean, args.std)
    # this should always dispatch to transforms_imagenet_train
    transform = transforms_imagenet_train_aug(
            img_size=args.input_size,
            color_jitter=args.color_jitter,
            interpolation=args.train_interpolation,
            auto_augment=args.aa,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            to_tensor=True,
            normalize=True,
            mean=mean,
            std=std
        )
    return transform

def build_prepoison_transform(args):
    resize_im = args.input_size > 32
    resized_dset = ['GTSRB', 'PubFig50']
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3, antialias=True),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    if args.data_set in resized_dset :
        t.append(transforms.Resize((args.input_size, args.input_size), antialias=True))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def build_eval_transform(args, to_tensor=False):
    t = []
    if to_tensor:
        t.append(transforms.ToTensor())

    if args.aug_method == 'simple':
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    elif args.aug_method == 'pm1':
        t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_dataset(args, is_train, prepoison_transform=None):
    if args.data_set == 'CIFAR10':
        print(args.data_path)
        dataset = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=prepoison_transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=prepoison_transform)
        nb_classes = 100                             
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=prepoison_transform)
        nb_classes = 1000
    elif args.data_set == 'T-IMNET':
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'tiny-imagenet-200', 'train' if is_train else 'val'), # test?
            transform=prepoison_transform)
        nb_classes = 200
    elif args.data_set == 'GTSRB':
        dataset = datasets.GTSRB(args.data_path, 'train' if is_train else 'test', transform=prepoison_transform)
        dataset.targets = torch.tensor([i[1] for i in dataset._samples]).squeeze()
        nb_classes = 43
    elif args.data_set == 'CELEBATTR':
        dataset = CelebA_attr(args.data_path, is_train, transform=prepoison_transform)
        dataset.targets = torch.load('resources/celeba_label.pth')
        nb_classes = 8
    elif args.data_set == 'IMAGEWOOF':
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'imagewoof2-160', 'train' if is_train else 'val'), # test?
            transform=prepoison_transform)
        nb_classes = 10
    else:
        raise Exception(f'Unsupported dataset: {args.data_set}')
    return dataset, nb_classes

def build_backdoor_dataset(attack_portion, args, prepoison_transform=None, transform=None, return_orig=True, selected_idx=None, is_train=True, no_aug=False):
    if transform is None:
        if not is_train or no_aug:
            # transform = build_eval_transform(args)
            transform = build_eval_transform(args, to_tensor=False)
        elif args.aug_method == 'simple':
            transform = build_simple_aug_transform(args)
        elif args.aug_method == 'pm1':
            transform = build_pm1_aug_transform(args)
        else:
            transform = build_aug_transform(args)

    if args.verbose >= 2:
        print('train_aug_transform: ', transform)
    attack_target_transform = get_target_transform(args)
    

    if prepoison_transform is None:
        prepoison_transform = build_prepoison_transform(args)

    if args.verbose >= 2:
        print('train_prepoison_transform: ', prepoison_transform)

    dataset, nb_classes = get_dataset(args, is_train, prepoison_transform)

    if args.attack_type in ['badnet', None]:
        dataset = PoisonedDataset(dataset, attack_target_transform, 
                                    portion=attack_portion,
                                    transform=transform,
                                    poisoned_pixel_val=args.attack_pixel_val,
                                    pattern_width=args.attack_pattern_width,
                                    return_orig=return_orig,
                                    selected_idx=selected_idx,
                                    clean_label=args.attack_mode == 'clean_label', train=is_train, target_cls=args.attack_label, trigger_type=args.badnet_trigger
                                    )

    elif args.attack_type == 'wanet':
        dataset = WaNetPoisonedDataset(dataset, attack_target_transform, 
                                    portion=attack_portion,
                                    transform=transform, input_height=args.input_size, noise_grid=args.noise_grid, identity_grid=args.identity_grid, return_orig=return_orig)
    elif args.attack_type == 'blended':
        dataset = BlendedPoisonedDataset(dataset, attack_target_transform, 
                                    portion=attack_portion,
                                    transform=transform,
                                    selected_idx=selected_idx,
                                    clean_label=args.attack_mode == 'clean_label', train=is_train, target_cls=args.attack_label, input_size=args.input_size, blended_rate=args.blended_rate, random_seed=args.seed)
    elif args.attack_type == 'SIG':
        dataset = SIGPoisonedDataset(dataset, attack_target_transform, 
                                    portion=attack_portion,
                                    transform=transform,
                                    selected_idx=selected_idx, return_orig=return_orig,
                                    clean_label=args.attack_mode == 'clean_label', train=is_train, target_cls=args.attack_label, input_size=args.input_size,
                                    delta=args.sig_delta, f=args.sig_f, random_seed=args.seed)

    print(type(dataset))
    return dataset, nb_classes
