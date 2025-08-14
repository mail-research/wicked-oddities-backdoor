from argparse import ArgumentParser

import torch
import numpy as np

from defenses.neural_cleanse import neural_cleanse
from defenses.fine_pruning import fine_pruning
from defenses.strip import strip
from defenses.cognitive_distillation import cognitive_distillation
from defenses.frequency import frequency_detect
from defenses.ss import ss
from defenses.ac import ac
from defenses.spectre import run_spectre
from defenses.channel_lipschitz import channel_lipschitz_defense
from defenses.rnp import RNP
from defenses.ft_sam import FT_SAM
from defenses.ibd_psc import run_ibd_psc
from defenses.asset import run_asset
from defenses.scale_up import run_scale_up

def parse_args():
    parser = ArgumentParser('Defense script', add_help=False)

    # General
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=32, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', default=None, type=str, required=True, help='name of model')
    parser.add_argument('--checkpoint', default=None, type=str, required=True, help='path for model checkpoint')
    parser.add_argument('--output_dir', default='', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Defense strategy
    parser.add_argument('--defense', type=str, default='NC', required=False, 
                        choices=['NC', 'STRIP', 'FP', 'CD', 'FREQ', 'SS', 'AC', 'SPECTRE', 'CLP', 'RNP', 'FT-SAM', 'IBD_PSC', 'ASSET', 'SCALE_UP'])

    # Dataset params
    parser.add_argument('--data-path', default='~/data', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', 
                        choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'CELEBATTR', 'T-IMNET', 'IMNET', 'INAT', 'INAT19', 'MNIST'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--aug_method', default='simple', type=str, help='Augmentation method')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained model')
    
    # Backdoor params
    parser.add_argument('--attack_type', default=None, type=str)
    parser.add_argument('--attack_mode', default='clean_label', type=str)
    parser.add_argument('--attack_label', default=0, type=int)
    parser.add_argument('--attack_pixel_val', default=1, type=float)
    parser.add_argument('--attack_pattern_width', default=3, type=int)
    parser.add_argument('--clean-acc-tol', default=0.1, type=float)
    parser.add_argument('--compression_type', default='JPEG', type=str)
    parser.add_argument('--compression_method', default='PIL', choices=['PIL', 'CV2'], type=str)
    parser.add_argument('--compression_quality', default=70, type=int)
    parser.add_argument('--badnet_trigger', default='hard', type=str)

    parser.add_argument('--blended_rate', default=0.2, type=float)
    parser.add_argument('--sig_delta', default=20, type=float)
    parser.add_argument('--sig_f', default=6, type=float)

    # DDP params
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # STRIP params
    parser.add_argument('--strip_n_sample', default=100, type=int, help='Number of samples for STRIP')
    parser.add_argument('--strip_test_rounds', default=10, type=int, help='Test rounds for STRIP')
    parser.add_argument("--strip_n_test", type=int, default=100)
    parser.add_argument('--strip_detection_boundary', default=0.2, type=float, help='Detection boundary') # According to original paper

    # Neural Cleanse params
    parser.add_argument('--nc_lr', type=float, default=0.1, help='Neural Cleanse Learning Rate')
    parser.add_argument("--nc_init_cost", type=float, default=1e-3)
    parser.add_argument("--nc_atk_succ_threshold", type=float, default=98.0)
    parser.add_argument("--nc_early_stop", type=bool, default=True)
    parser.add_argument("--nc_early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--nc_early_stop_patience", type=int, default=25)
    parser.add_argument("--nc_patience", type=int, default=5)
    parser.add_argument("--nc_cost_multiplier", type=float, default=2)
    parser.add_argument("--nc_epoch", type=int, default=20)
    parser.add_argument("--nc_target_label", type=int)
    parser.add_argument("--nc_total_label", type=int)
    parser.add_argument("--nc_EPSILON", type=float, default=1e-7)
    parser.add_argument("--nc_n_times_test", type=int, default=10)
    parser.add_argument("--nc_use_norm", type=int, default=1)

    # Cognitive Distillation params
    parser.add_argument('--cd_lr', type=float, default=0.1, help='Cognitive Distillation lr')
    parser.add_argument('--cd_p', type=int, default=1)
    parser.add_argument('--cd_gamma', type=float, default=0.01)
    parser.add_argument('--cd_beta', type=float, default=1.0)
    parser.add_argument('--cd_num_steps', type=int, default=100)
    parser.add_argument('--cd_mask_channel', type=int, default=1)
    parser.add_argument('--cd_norm_only', action='store_true', default=False)

    # Frequency Detection params
    parser.add_argument('--freq_detector_checkpoint', type=str, default=None)
    
    parser.add_argument('--strategy', type=str, default='none')
    parser.add_argument('--attack_portion', type=float, default=0.1)
    parser.add_argument("--surrogate_model", type=str, default='resnet18')

    parser.add_argument('--verbose', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.output_dir = f'defense_log_new/{args.model}_' + args.output_dir
    # state_dict = torch.load(args.checkpoint)
    # args.mean, args.std = state_dict['args'].mean, state_dict['args'].std
    print(args)
    if args.defense == 'NC':
        neural_cleanse(args)
    elif args.defense == 'STRIP':
        strip(args)
    elif args.defense == 'FP':
        fine_pruning(args)
    elif args.defense == 'CD':
        cognitive_distillation(args)
    elif args.defense == 'FREQ':
        frequency_detect(args)
    elif args.defense == 'SS':
        ss(args)
    elif args.defense == 'AC':
        ac(args)
    elif args.defense == 'SPECTRE':
        run_spectre(args)
    elif args.defense == 'CLP':
        channel_lipschitz_defense(args)
    elif args.defense == 'RNP':
        RNP(args)
    elif args.defense == 'FT-SAM':
        FT_SAM(args)
    elif args.defense == 'IBD_PSC':
        run_ibd_psc(args)
    elif args.defense == 'ASSET':
        run_asset(args)
    elif args.defense == 'SCALE_UP':
        run_scale_up(args)
    