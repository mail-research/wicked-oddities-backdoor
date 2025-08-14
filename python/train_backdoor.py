import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import pickle

from pathlib import Path

import timm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from poisoned_datasets import build_backdoor_dataset, build_eval_transform
from datasets import build_dataset

from engine import train_one_epoch_bd, evaluate_bd
from losses import DistillationLoss
from samplers import RASampler
import utils
from ulib.submit import _create_run_dir_local
from ulib.external_logger import ExternalLogger 


def get_none_type(s):
    return None if s.lower() in [None, '', 'none'] else s

def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--basedir", default='exp', type=str)
    parser.add_argument("--external_logger", default=None, type=str)
    parser.add_argument("--external_logger_args", default=None, type=str)
    parser.add_argument("--elabel", default=None, type=str)
    parser.add_argument("--run_name", type=str, default=None)

    # Model parameters
    parser.add_argument('--model', default='myresnet18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    parser.add_argument('--trainable_layers', default=None, type=str, metavar='TRAINABLE',
                        help='prefixes of trainable layers')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora-layer', nargs='+', type=int, help='LORA layer index')
    parser.add_argument('--ft-lc', action='store_true')
    parser.add_argument('--freeze-layer', type=int, default=-1)

    # RVT params
    parser.add_argument('--use_patch_aug', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=int, default=30, metavar='N', nargs='+',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--aug-method', type=str, default=None,
                        help='Choose the aug-method to use, default to timm (None)')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained model')

    # Dataset parameters
    parser.add_argument('--data-path', default='~/data', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', 
                        choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'CELEBATTR', 'T-IMNET', 'IMNET', 'INAT', 'INAT19', 'IMAGEWOOF', 'PubFig', 'PubFig50'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # eval parameters
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--inc_path', default=None, type=str, help='imagenet-c')
    parser.add_argument('--ina_path', default=None, type=str, help='imagenet-a')
    parser.add_argument('--inr_path', default=None, type=str, help='imagenet-r')
    parser.add_argument('--insk_path', default=None, type=str, help='imagenet-sketch')
    parser.add_argument('--fgsm_test', action='store_true', default=False, help='test on FGSM attacker')
    parser.add_argument('--pgd_test', action='store_true', default=False, help='test on PGD attacker')

    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Backdoor params
    parser.add_argument('--attack_type', default=None, type=str)
    parser.add_argument('--attack_portion', default=0.2, type=float)
    parser.add_argument('--attack_mode', default='all2one', type=str)
    parser.add_argument('--attack_label', default=0, type=int)
    parser.add_argument('--attack_pixel_val', default=2.640000104904175, type=float)
    parser.add_argument('--attack_pattern_width', default=2, type=int)
    parser.add_argument('--badnet_trigger', default='easy', type=str)
    parser.add_argument('--clean-acc-tol', default=0.1, type=float)
    parser.add_argument('--clean-label', action='store_true', help='clean label attack')

    parser.add_argument('--blended_rate', default=0.2, type=float)
    parser.add_argument('--sig_delta', default=20, type=float)
    parser.add_argument('--sig_f', default=6, type=float)
    
    
    # ConvNext Params
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)
    
    # selection
    parser.add_argument("--selection_strategy", type=get_none_type, default=None)
    parser.add_argument("--surrogate_model", type=str, default='resnet18')
    parser.add_argument("--percentile", type=int, default=-1)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--subset_size", default=-1, type=int)
    # save checkpoint
    parser.add_argument("--save_ckpt", action='store_false', help='save model and optim checkpoints')
    return parser

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def log(s, verbose=0, level=0, flush=True):
    if level >= verbose:
        print(s, flush=flush)
        
def freeze_layers(args, model):
    trainable_layers = []
    if args.trainable_layers:
        trainable_layers = args.trainable_layers.split('|')
        
    for n, p in model.named_parameters():
        if trainable_layers and np.sum([n.startswith(tr_layer_prefix) for tr_layer_prefix in trainable_layers]) == 0:
            p.requires_grad = False
        log(f'{n}: {p.requires_grad}', args.verbose, 2)

def create_model(args):
    mean, std = None, None
    if args.model.startswith('my'):
        if args.model == 'mypreactresnet18':
            from models.preact_resnet import PreActResNet18
            model = PreActResNet18(num_classes=args.nb_classes)
        elif args.model == 'myresnet18':
            from models.resnet import ResNet18
            model = ResNet18(num_classes=args.nb_classes, input_size=args.input_size)
    else:
        if 'vit' in args.model or 'deit' in args.model:
            model = timm.models.create_model(
                args.model,
                img_size=args.input_size,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None
            )
            mean, std = model.default_cfg['mean'], model.default_cfg['std']

            if args.ft_lc:
                for name, param in model.named_parameters():
                    if name[:4] != 'head':
                        param.requires_grad = False
            if args.freeze_layer >= 0:
                def set_module_grad(m, trainable=True):
                    for p in m.parameters():
                        p.requires_grad = trainable
                    return m
                
                model = set_module_grad(model, False)
                for i in range(len(model.blocks)):
                    if i >= args.freeze_layer:
                        model.blocks[i] = set_module_grad(model.blocks[i])
                model.reset_classifier(args.nb_classes)

        elif 'vgg' in args.model:
            model = timm.models.create_model(
                args.model,
                #img_size=args.input_size,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
            )
            mean, std = model.default_cfg['mean'], model.default_cfg['std']
        elif 'convnext' in args.model:
            model = timm.models.create_model(
                args.model, 
                pretrained=False, 
                num_classes=args.nb_classes, 
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
            )            
            mean, std = model.default_cfg['mean'], model.default_cfg['std']
        else:
            model = timm.models.create_model(
                args.model,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None
            )
            mean, std = model.default_cfg['mean'], model.default_cfg['std']
    return model, mean, std

        
def main(args):
    if not args.output_dir:
        if args.attack_type is not None:
            args.basedir = os.path.join(args.basedir, args.attack_type)

        dir_suffix = f'lr{args.lr}-bsize{args.batch_size}-opt{args.opt}-sched{args.sched}'
        if args.attack_mode == 'all2all':
            args.output_dir = _create_run_dir_local(
                os.path.join(args.basedir, args.data_set, args.attack_mode, args.model), dir_suffix)
        elif args.attack_mode == 'all2one' or args.attack_mode == 'clean_label':
            args.output_dir = _create_run_dir_local(
                os.path.join(args.basedir, args.data_set, f'{args.attack_mode}_{args.attack_label}', args.model), dir_suffix)

    print(f'Working dir is {args.output_dir}') 
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('| distributed init {}(rank {})'.format(
                args.world_size, args.rank), flush=True)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        print('Not using distributed mode')
        args.distributed = False

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    class_dict = {'CIFAR10': 10, 'GTSRB': 43, 'T-IMNET': 200, 'CIFAR100': 100, 'CELEBATTR': 8, 'IMAGEWOOF': 10, 'PubFig': 150, 'PubFig50': 50}
    args.nb_classes = class_dict[args.data_set]
    print(f"Creating model: {args.model}:pretrained={args.pretrained}:nb_classes={args.nb_classes}")

    model, mean, std = create_model(args)
    args.mean, args.std = mean, std
    log(model, args.verbose, 2)        
    freeze_layers(args, model)

    model.to(device)
    if args.attack_type == 'wanet':
        k = 4
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        import torch.nn.functional as F
        noise_grid = (
            F.upsample(ins, size=args.input_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=args.input_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        args.noise_grid = noise_grid
        args.identity_grid = identity_grid
    selected_idx = None

    dset, _ = build_dataset(True, args)
    if args.data_set == 'CIFAR10':
        label_list = torch.tensor(dset.targets)
    elif args.data_set == 'GTSRB':
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

        selected_idx = (label_list == args.attack_label).nonzero().squeeze()[hard_sample].cpu().numpy().squeeze()
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
        selected_idx = (label_list == args.attack_label).nonzero().squeeze()[hard_sample].cpu().numpy().squeeze()

    else:
        selected_idx = None
    
    dataset_train, args.nb_classes = build_backdoor_dataset(args.attack_portion, args=args, selected_idx=selected_idx, is_train=True)
    args.num_bd = dataset_train.num_bd
    dataset_val, _ = build_backdoor_dataset(1.0, args=args, is_train=False)


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, #sampler=sampler_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_train.collate_fn if args.attack_type == 'wanet1' else None
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, #sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_val.collate_fn_eval if args.attack_type == 'wanet1' else None
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {:0.4f}M'.format(n_parameters*1.0/1e6))


    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    
    if args.sched == 'warmup_cosine':
        from scheduler import WarmupCosineSchedule
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=5, t_total=args.epochs) #step is now epoch
    elif args.sched == 'multistep_torch':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, args.decay_rate)
    elif args.sched == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        lr_scheduler, _ = create_scheduler(args, optimizer)
    
    log(f'Using optimizer: {optimizer}', args.verbose, 2)
    log(f'Using scheduler: {args.sched}:{lr_scheduler}', args.verbose, 2)


    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    log(f'Using criterion: {criterion}', args.verbose, 2)

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:

        test_transform = build_eval_transform(args)
        test_stats = evaluate_bd(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: clean {test_stats['clean_acc1']:.1f}% poison {test_stats['poison_acc1']:.1f}%")

        return
    
    external_logger = ExternalLogger(args)
    if args.external_logger == 'neptune':
        external_logger.set_dict(args.__dict__)
        external_logger.set_val('args', args)

    pickle.dump(args, open(os.path.join(args.output_dir, 'params.pkl'), 'wb'))
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_clean_accuracy = 0.0
    max_poison_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_bd(
            args, model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        if args.output_dir and args.save_ckpt:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema else None,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate_bd(data_loader_val, model, device)
                          
        print(f"Accuracy of the network on the {len(dataset_val)} test images: clean {test_stats['clean_acc1']:.1f}% poison {test_stats['poison_acc1']:.1f}%")
              
        if test_stats["clean_acc1"] > max_clean_accuracy \
              or (test_stats["clean_acc1"] > max_clean_accuracy-args.clean_acc_tol and  test_stats["poison_acc1"] > max_poison_accuracy):
              
            max_clean_accuracy = test_stats["clean_acc1"]
            max_poison_accuracy = test_stats["poison_acc1"]

            if args.output_dir:
                best_model_paths = [output_dir / 'best_model.pth']
                for checkpoint_path in best_model_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema) if model_ema else None,
                        'clean_accuracy': max_clean_accuracy,
                        'poison_accuracy': max_poison_accuracy,
                        'args': args,
                    }, checkpoint_path)
              
        print(f'Max accuracy: clean {max_clean_accuracy:.2f}% poison {max_poison_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'best_poison_accuracy': max_poison_accuracy,
                     'best_clean_accuracy': max_clean_accuracy,
                    }
        
        external_logger.log_dict(log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
