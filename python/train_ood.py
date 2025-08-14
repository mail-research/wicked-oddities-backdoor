import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet_ssl import *
from poisoned_datasets import (
    build_extra_dataset, build_eval_transform, build_simple_aug_transform
)
from train_backdoor import get_args_parser, create_model
from torch.utils.data import DataLoader
from torchvision import transforms

def parse_args():
    parser = get_args_parser()
    parser.add_argument('--target', type=int, default=1, help='Target class')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes for OOD')
    parser.add_argument('--save_path', type=str, default='resources/ood_model/', help='Path to save the model')
    return parser.parse_args()

def initialize_model(args):
    args.nb_classes = args.num_classes + 1
    model, mean, std = create_model(args)
    args.mean, args.std = mean, std
    return model

def get_datasets(args):
    pre_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])
    transform = transforms.Compose([pre_transform, build_simple_aug_transform(args)])
    test_transform = transforms.Compose([pre_transform, build_eval_transform(args)])
    
    dataset_train = build_extra_dataset(args, args.target, True, transform, num_classes=args.num_classes)
    dataset_test = build_extra_dataset(args, args.target, False, test_transform, num_classes=args.num_classes)
    return dataset_train, dataset_test

@torch.no_grad()
def compute_acc(model, loader, device):
    model.eval()
    corr, total = 0, 0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        pred = model(img).argmax(1)
        corr += (pred == label).sum()
        total += img.shape[0]
    return corr / total

def train(args):
    # Initialize model, datasets, and loaders
    model = initialize_model(args)
    model.to(args.device)
    dataset_train, dataset_test = get_datasets(args)
    loader = DataLoader(dataset_train, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training and evaluation
    best_acc = 0
    save_path = os.path.join(args.save_path, f"{args.data_set}_{args.target}_{args.num_classes}/")
    os.makedirs(save_path, exist_ok=True)

    for epoch in (pbar := tqdm(range(args.epochs))):
        model.train()
        for img, label in loader:
            img, label = img.to(args.device), label.to(args.device)
            out = model(img)
            loss = nn.functional.cross_entropy(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        # Evaluate and save model
        acc = compute_acc(model, test_loader, args.device)
        pbar.set_description(f'Epoch: {epoch}, Acc: {acc.item():.4f}')
        torch.save(model.state_dict(), os.path.join(save_path, 'last.pth'))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()