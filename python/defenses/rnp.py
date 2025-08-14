import torch.nn as nn
import torch
from tqdm import tqdm
import models
from poisoned_datasets import build_backdoor_dataset
from collections import OrderedDict
import pandas as pd
import numpy as np

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

def train_step_unlearning(model, criterion, optimizer, data_loader, device):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels, _, _) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def train_step_recovering(unlearned_model, criterion, mask_opt, data_loader, device):
    unlearned_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels, _, _) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        loss = criterion(output, labels)
        loss = 0.2 * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader, device):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in tqdm(thresholds):
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader, device=device)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader, device=device)

        # results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
        #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append([po_acc, cl_acc])
    return results

def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

def test(model, criterion, data_loader, device):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def RNP(args):
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
    clean_train_dataset, _ = build_backdoor_dataset(0, args, is_train=True)

    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)
    bd_loader = torch.utils.data.DataLoader(bd_dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)
    
    mode = args.attack_mode

    if args.model == 'myresnet18':
        from models.resnet import ResNet18, ResNet
        model = ResNet18(num_classes=args.nb_classes).to(args.device)
    model.load_state_dict(state_dict["model"])
    criterion = torch.nn.CrossEntropyLoss()
    print(test(model, criterion=criterion, data_loader=clean_loader, device=args.device))
    print(test(model, criterion=criterion, data_loader=bd_loader, device=args.device))


    unlearn_dset, _ = torch.utils.data.random_split(clean_train_dataset, [0.01, 0.99])
    # breakpoint()
    unlearn_loader = torch.utils.data.DataLoader(unlearn_dset, batch_size=128, num_workers=args.num_workers, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    for epoch in range(0, 21):
        train_loss, train_acc = train_step_unlearning(model, criterion=criterion, optimizer=optimizer,
                                      data_loader=unlearn_loader, device=args.device)

        scheduler.step()

        if train_acc <= 0.2:
            break
    
    from models.mask_batchnorm import MaskBatchNorm2d
    unlearned_model = ResNet18(num_classes=args.nb_classes, norm_layer=MaskBatchNorm2d).to(args.device)
    load_state_dict(unlearned_model, model.state_dict())

    parameters = list(unlearned_model.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=0.2, momentum=0.9)

    for epoch in range(1, 21):
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step_recovering(unlearned_model=unlearned_model, criterion=criterion, data_loader=unlearn_loader,
                                           mask_opt=mask_optimizer, device=args.device)

    save_mask_scores(unlearned_model.state_dict(), 'mask_values.txt')

    model = ResNet18(num_classes=args.nb_classes).to(args.device)
    model.load_state_dict(state_dict["model"])

    mask_values = read_data('mask_values.txt')
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))

    results = evaluate_by_threshold(
            model, mask_values, pruning_max=0.9, pruning_step=0.05,
            criterion=criterion, clean_loader=clean_loader, poison_loader=bd_loader, device=args.device
        )

    print(results)
        
