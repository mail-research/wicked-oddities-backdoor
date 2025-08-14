import torch
import torch.nn as nn

from poisoned_datasets import *
import matplotlib.pyplot as plt

import torch

def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        self.mean = torch.mean(data).item()
        self.std = torch.std(data).item()
        return

    def predict(self, data, t=1):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        p = (self.mean - data) / self.std
        p = torch.where((p > t) & (p > 0), 1, 0)
        return p.numpy()

    def analysis(self, data, is_test=False):
        """
            data (torch.tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (torch.tensor)
        """
        if self.norm_only:
            if len(data.shape) > 1:
                data = torch.norm(data, dim=[1, 2, 3], p=1)
            score = data
        else:
            score = torch.norm(data, dim=[1, 2, 3], p=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD

def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class CognitiveDistillation(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = True
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            features, logits = model(images, return_features=True)
        else:
            logits = model(images).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * torch.rand(b, c, 1, 1).to(images.device)
            if self.get_features:
                adv_fe, adv_logits = model(x_adv, return_features=True)
                # print(adv_fe.shape, features.shape)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()
    

class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def cognitive_distillation(args):
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

    if args.data_set == 'CIFAR10' or args.data_set == 'GTSRB' or args.data_set == 'CELEBATTR':
        denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    elif args.data_set == 'MNIST':
        denormalizer = Denormalize(args, [0.5], [0.5])
    else:
        denormalizer = Denormalize(args, [0, 0, 0], [1, 1, 1])

    # clean_dataset, _ = build_eval_dataset(0, args) # Attack portion should be 0 for for clean dataset
    # bd_dataset, _ = build_eval_dataset(1.0, args)
    args.mean, args.std = None, None
    train = True

    if args.data_set == 'CIFAR10':
        dset = datasets.CIFAR10(args.data_path, train=True)
        label_list = torch.tensor(dset.targets)
    elif args.data_set == 'GTSRB':
        dset = datasets.GTSRB(args.data_path, 'train')
        label_list = torch.tensor([i[1] for i in dset._samples]).squeeze()

    if args.selection_strategy == 'knn' and args.attack_mode == 'clean_label':
        
        total_samples = (torch.tensor(label_list, dtype=int) == args.attack_label).sum().item()
        num_bd = int(args.attack_portion * total_samples)
        knn_score =  - torch.load(f'resources/knn_{args.data_set}_{args.attack_label}_{args.surrogate_model}.pth')

        if args.percentile >= 0:
            start_id = int(total_samples * args.percentile / 100)
            hard_sample = knn_score.sort()[1][start_id:start_id+num_bd]
        else:
            hard_sample = knn_score.sort()[1][-num_bd:]

        poison_indices = (label_list == args.attack_label).nonzero().squeeze()[hard_sample].cpu().numpy().squeeze()
    elif 'loss_ood' in args.selection_strategy and args.attack_mode == 'clean_label':

        total_samples = (torch.tensor(label_list, dtype=int) == args.attack_label).sum().item()
        num_bd = int(args.attack_portion * total_samples)
        score =  torch.load(f'resources/{args.selection_strategy}_{args.data_set}_{args.attack_label}_{args.surrogate_model}.pth')
        if args.percentile >= 0:
            start_id = int(total_samples * args.percentile / 100)
            hard_sample = score.sort()[1][start_id:start_id+num_bd]
        else:
            hard_sample = score.sort()[1][-num_bd:]
        hard_sample = score.sort()[1][-num_bd:]
        poison_indices = (label_list == args.attack_label).nonzero().squeeze()[hard_sample].cpu().numpy().squeeze()

    else:
        poison_indices = None

    bd_dataset, _ = build_backdoor_dataset(0.1, args, is_train=train, selected_idx=poison_indices, no_aug=True)
    nonpoison_indices = set(range(50000)) - set(poison_indices)
    clean_dataset = torch.utils.data.Subset(bd_dataset, list(nonpoison_indices))
    bd_dataset = torch.utils.data.Subset(bd_dataset, poison_indices)

    clean_test_dataset, _ = build_backdoor_dataset(0, args, is_train=False, no_aug=True)
    bd_test_dataset, _ = build_backdoor_dataset(1, args, is_train=False, no_aug=True)

    if args.model == 'myresnet18':
        from models.resnet import ResNet18
        model = ResNet18(num_classes=args.nb_classes)
    elif args.model == 'mypreactresnet18':
        from models.preact_resnet import PreActResNet18
        model = PreActResNet18(num_classes=args.nb_classes)
    elif args.model == 'mymnistnet':
        from models.mnist_net import MNISTNet
        model = MNISTNet()

    mode = 'attack' if args.attack_mode == 'all2all'or args.attack_mode == 'all2one' else 'clean'
    # args.checkpoint = os.path.join(args.checkpoint, 'best_model.pth')

    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['model'])
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)

    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    bd_loader = torch.utils.data.DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    bd_test_loader = torch.utils.data.DataLoader(
        bd_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )


    detector = CognitiveDistillation(args.cd_lr, args.cd_p, args.cd_gamma, args.cd_beta, num_steps=args.cd_num_steps, 
                                mask_channel=args.cd_mask_channel, norm_only=args.cd_norm_only)
    detector.to(args.device)
    analysis = CognitiveDistillationAnalysis()

    def get_masks(loader):
        masks = []
        for idx, (input, target, _, _) in enumerate(tqdm(loader)):
            input = input.to(args.device)
            target = target.to(args.device)
            mask = detector(model, input, target)
            masks.append(mask)
        masks = torch.cat(masks)
        return masks


    clean_masks = get_masks(clean_loader)
    bd_masks = get_masks(bd_loader)

    clean_test_masks = get_masks(clean_test_loader)
    bd_test_masks = get_masks(bd_test_loader)
    all_masks = torch.cat([clean_masks, bd_masks])
    all_test_masks = torch.cat([clean_test_masks, bd_test_masks])
    analysis.train(all_masks)
    scores = analysis.analysis(all_masks)
    test_scores = analysis.analysis(all_test_masks)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score([0] * len(clean_masks) + [1] * len(bd_masks), scores)
    test_auc = roc_auc_score([0] * len(clean_test_masks) + [1] * len(bd_test_masks), test_scores)
    # auc = roc_auc_score([0] * len(clean_score) + [1] * len(bd_score), torch.cat([clean_score, bd_score]))
    print(auc, test_auc)
    exit()
    breakpoint()
    with open(l1_norm_txt, "w") as file:
        cln = ' '.join(str(a.item()) for a in clean_l1_norms)
        bln = ' '.join(str(a.item()) for a in bd_l1_norms)
        file.write(cln)
        file.write('\n')
        file.write(bln)
        file.close()

    plt.hist(clean_l1_norms, alpha=0.5, label='Clean', bins=100, color='blue')
    plt.hist(bd_l1_norms, alpha=0.5, label='Backdoor', bins=100, color='red')
    if args.data_set == 'T-IMNET':
        plt.title(f'Tiny ImageNet')
    elif args.data_set == 'CELEBATTR':
        plt.title(f'CelebA')
    else:
        plt.title(f'{args.data_set}')
    plt.legend()
    plt.xlabel('L1 norm of the mask')
    plt.ylabel('Number of samples')

    l1_norm_path = os.path.join(args.output_dir, f'l1_norms.png')
    plt.savefig(l1_norm_path)
    plt.savefig(os.path.join(args.output_dir, f'l1_norms.pdf'))

    clean_mask_path = os.path.join(args.output_dir, f'clean_masks')
    bd_mask_path = os.path.join(args.output_dir, f'bd_masks')

    clean_img_path = os.path.join(args.output_dir, f'clean_img')
    bd_img_path = os.path.join(args.output_dir, f'bd_img')

    os.makedirs(clean_mask_path, exist_ok=True)
    os.makedirs(bd_mask_path, exist_ok=True)
    os.makedirs(clean_img_path, exist_ok=True)
    os.makedirs(bd_img_path, exist_ok=True)

    n_images = 5

    for idx, (clean_mask, bd_mask) in enumerate(zip(clean_masks, bd_masks)):
        if idx == n_images:
            break
        plt.imsave(os.path.join(clean_mask_path, f'clean_mask_{idx+1}.png'), clean_mask.cpu().numpy(), cmap='gray')
        plt.imsave(os.path.join(bd_mask_path, f'bd_mask_{idx+1}.png'), bd_mask.cpu().numpy(), cmap='gray')

        ci = clean_dataset[idx][0].permute(1, 2, 0).cpu()
        bi = bd_dataset[idx][0].permute(1, 2, 0).cpu()
        ci = denormalizer(ci).numpy()
        bi =  denormalizer(bi).numpy()

        plt.imsave(os.path.join(clean_img_path, f'clean_img_{idx+1}.png'), ci)
        plt.imsave(os.path.join(bd_img_path, f'bd_img_{idx+1}.png'), bi)
    
    print(f'Output saved at {args.output_path}!')
    

