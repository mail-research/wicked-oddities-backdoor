import os
import torch
import timm
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, GTSRB, ImageFolder
from poisoned_datasets import build_extra_dataset, CelebA_attr
from models.resnet_ssl import resnet50
from train_backdoor import create_model
import argparse

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

def parse_args():
    parser = argparse.ArgumentParser(description="Select data strategy")
    parser.add_argument('--strategy', type=str, default='knn', choices=['knn', 'mean', 'loss_ood'], help='Strategy to use')
    parser.add_argument('--model_name', type=str, default='vicreg', choices=['vicreg', 'resnet50'], help='Model name')
    parser.add_argument('--dset', type=str, default='CIFAR10', choices=['CIFAR10', 'GTSRB', 'CELEBATTR', 'IMAGEWOOF'], help='Dataset name')
    parser.add_argument('--target', type=int, default=2, help='Target class')
    parser.add_argument('--data_path', type=str, default='~/data', help='Path to dataset')
    parser.add_argument('--ood_classes', type=int, default=100, help='Number of OOD classes')
    parser.add_argument('--k', type=int, default=50, help='Number of neighbors for KNN')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., "cuda:0" or "cpu")')
    return parser.parse_args()

def initialize_model(args):
    if args.model_name == 'vicreg':
        model, _ = resnet50()
        model.load_state_dict(torch.load('pretrained/resnet50_vicreg.pth'))
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        model.forward_features = model.forward
    elif args.model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
    elif args.strategy == 'loss_ood':
        args.model = 'myresnet18'
        args.nb_classes = args.ood_classes + 1
        model, mean, std = create_model(args)
        if mean is None:
            mean, std = CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
        model.load_state_dict(torch.load(f'resources/ood_model/{args.dset}_{args.target}/best.pth'))
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    
    model = model.to(args.device)
    return model, mean, std

def get_dataset(args, transform):
    if 'ood' in args.strategy:
        return build_extra_dataset(args, args.target, True, transform, num_classes=args.ood_classes)
    elif args.dset == 'CIFAR10':
        return CIFAR10(root=args.data_path, train=True, transform=transform)
    elif args.dset == 'GTSRB':
        return GTSRB(args.data_path, 'train', transform=transform)
    elif args.dset == 'CELEBATTR':
        return CelebA_attr(args.data_path, True, transform=transform)
    elif args.dset == 'IMAGEWOOF':
        return ImageFolder(os.path.join(args.data_path, 'imagewoof2-160', 'train'), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dset}")

def loss_ood_selection(args, model, loader):
    target_class = args.ood_classes
    total, corr = 0, 0
    target_loss_list = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            mask = label == target_class
            if mask.sum() > 0:
                img, label = img.to(args.device), label.to(args.device)
                target_label = torch.ones_like(label) * target_class
                logits = model(img)
                target_loss = torch.nn.functional.cross_entropy(logits, target_label, reduction='none')
                total += img.shape[0]
                corr += (logits.argmax(1) == label).sum()
                target_loss_list.append(target_loss[label == target_class])
        target_loss_list = torch.cat(target_loss_list)
        torch.save(target_loss_list.cpu(), f'resources/loss_ood_{args.ood_classes}_{args.dset}_{args.target}_{args.model_name}.pth')

def pretrained_selection(args, model, loader):
    feat = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            if (label == args.target).sum() > 0:
                img = img[label == args.target].to(args.device)
                h_feat = model.forward_features(img) if args.model_name != 'vicreg' else model(img)
                feat.append(h_feat)
    feat = torch.cat(feat)

    if args.strategy == 'knn':
        feat /= feat.norm(2, dim=1, keepdim=True)
        score = feat @ feat.T
        score[range(len(score)), range(len(score))] = -1
        knn = [i.sort()[0][-args.k:].mean() for i in score]
        torch.save(torch.tensor(knn), f'resources/knn_{args.dset}_{args.target}_{args.model_name}_{args.k}.pth')
    elif args.strategy == 'mean':
        mean_feat = feat.mean(0)
        mean_feat /= mean_feat.norm(2)
        feat /= feat.norm(2, dim=1, keepdim=True)
        score = feat @ mean_feat
        torch.save(-score.cpu(), f'resources/mean_{args.dset}_{args.target}_{args.model_name}.pth')

def main():
    args = parse_args()
    model, mean, std = initialize_model(args)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train = get_dataset(args, transform)
    loader = DataLoader(dataset_train, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)

    if args.strategy == 'loss_ood':
        loss_ood_selection(args, model, loader)
    elif args.strategy in ['knn', 'mean']:
        pretrained_selection(args, model, loader)
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

if __name__ == "__main__":
    main()