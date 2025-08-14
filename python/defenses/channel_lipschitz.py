import torch.nn as nn
import torch
from tqdm import tqdm

from poisoned_datasets import build_backdoor_dataset

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)

@torch.no_grad()
def get_acc(model, loader, device):
    model.eval()
    corr, total = 0, 0
    for img, label, _, _ in tqdm(loader):
        img, label = img.to(device), label.to(device)
        out = model(img).argmax(1)
        corr += (out == label).sum()
        total += out.shape[0]

    return (corr / total).item()

def channel_lipschitz_defense(args):
    if args.data_set == 'CELEBATTR':
        args.nb_classes = 8
        args.input_size = 64
        args.input_channel = 3
    elif args.data_set == 'T-IMNET':
        args.nb_classes = 200
        args.input_size = 64
        args.input_channel = 3
    elif args.data_set == 'GTSRB':
        args.nb_classes = 43
        args.input_size = 32
        args.input_channel = 3
    elif args.data_set == 'CIFAR10':
        args.nb_classes = 10
        args.input_size = 32
        args.input_channel = 3
    elif args.data_set == 'MNIST':
        args.nb_classes = 10
        args.input_size = 28
        args.input_channel = 1

    args.nc_total_label = args.nb_classes
    state_dict = torch.load(args.checkpoint)
    args.mean, args.std = state_dict['args'].mean, state_dict['args'].std
    if args.mean is None:
         args.mean, args.std = CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD

    clean_dataset, _ = build_backdoor_dataset(0, args, is_train=False) # Attack portion should be 0 for for clean dataset
    bd_dataset, _ = build_backdoor_dataset(1.0, args, is_train=False)

    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)
    bd_loader = torch.utils.data.DataLoader(bd_dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)
    
    mode = args.attack_mode

    if args.model == 'myresnet18':
        from models.resnet import ResNet18, ResNet
        model = ResNet18(num_classes=args.nb_classes).to(args.device)
    model.load_state_dict(state_dict["model"])
    clean_acc = get_acc(model, clean_loader, args.device)
    asr = get_acc(model, bd_loader, args.device)
    print(f'Before prunning, Acc: {clean_acc}, ASR: {asr}')
    CLP(model, 3)

    clean_acc = get_acc(model, clean_loader, args.device)
    asr = get_acc(model, bd_loader, args.device)
    print(f'After prunning, Acc: {clean_acc}, ASR: {asr}')