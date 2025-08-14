import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.insert(0, "../..")

import os
# import matplotlib.pyplot as plt
import numpy as np

import timm
# from transformers import AutoModelForImageClassification
from poisoned_datasets import build_backdoor_dataset, build_eval_dataset

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class Normalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

def create_model(args):
    if args.model.startswith('my'):
        if args.model == 'mypreactresnet18':
            from models.preact_resnet import PreActResNet18
            model = PreActResNet18(num_classes=args.nb_classes)
        elif args.model == 'myresnet18':
            from models.resnet import ResNet18
            model = ResNet18(num_classes=args.nb_classes)
        elif args.model == 'mymnistnet':
            from models.mnist_net import MNISTNet
            model = MNISTNet()
    elif 'lora' in args.model:
        if 'vit' in args.model or 'deit' in args.model:
            model = AutoModelForImageClassification(
                args.model,
                num_labels=args.nb_classes,
                ignore_mismatch_sizes=True,
                image_size=args.input_size,
            )
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
        elif 'vgg' in args.model:
            model = timm.models.create_model(
                args.model,
                #img_size=args.input_size,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
            )
        elif 'convnext' in args.model:
            model = timm.models.create_model(
                args.model, 
                pretrained=False, 
                num_classes=args.nb_classes, 
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
            )            
        else:
            model = timm.models.create_model(
                args.model,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None
            )
    return model

class RegressionModel(nn.Module):
    def __init__(self, args, init_mask, init_pattern):
        self._EPSILON = args.nc_EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

        self.classifier = self._get_classifier(args)
        self.normalizer = self._get_normalize(args)
        self.denormalizer = self._get_denormalize(args)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, args):
        classifier = create_model(args)
        # Load pretrained classifie
        # ckpt_path = os.path.join(
        #     args.checkpoints, args.dataset, "{}_{}_morph.pth.tar".format(args.dataset, args.attack_mode)
        # )
        ckpt_path = args.checkpoint

        state_dict = torch.load(ckpt_path)
        
        classifier.load_state_dict(state_dict["model"])
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(args.device)

    def _get_denormalize(self, args):
        if args.data_set == "CIFAR10":
            denormalizer = Denormalize(args, args.mean, args.std)
        elif args.data_set == "MNIST":
            denormalizer = Denormalize(args, args.mean, args.std)
        elif args.data_set == "GTSRB" or args.data_set == "CELEBATTR" or args.data_set == 'T-IMNET':
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, args):
        if args.data_set == "CIFAR10":
            normalizer = Normalize(args, args.mean, args.std)
        elif args.data_set == "MNIST":
            normalizer = Normalize(args, [0.5], [0.5])
        elif args.data_set == "GTSRB" or args.data_set == "CELEBATTR" or args.data_set == 'T-IMNET':
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, args):
        super().__init__()

        # Best argsimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = args.nc_init_cost
        self.cost_multiplier_up = args.nc_cost_multiplier
        self.cost_multiplier_down = args.nc_cost_multiplier ** 1.5

    def reset_state(self, args):
        self.cost = args.nc_init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, args):
        result_dir = os.path.join(args.output_dir, args.data_set)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, args.attack_mode)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(args.nc_target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)


def train(args, init_mask, init_pattern):
    dataset_val, _ = build_backdoor_dataset(0, args=args, is_train=False) # attack portion should be 0, since WaNet didnt specify any attack portion
    sampler_train = torch.utils.data.RandomSampler(dataset_val)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Build regression model
    regression_model = RegressionModel(args, init_mask, init_pattern).to(args.device)

    # Set argsimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=args.nc_lr, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(args)

    for epoch in range(args.nc_epoch):
        early_stop = train_step(regression_model, optimizerR, test_dataloader, recorder, epoch, args)
        if early_stop:
            break

    # Save result to dir
    recorder.save_result_to_dir(args)

    return recorder, args


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, args):
    print("Epoch {} - Label: {} | {} - {}:".format(epoch, args.nc_target_label, args.data_set, args.attack_mode))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (_, _, inputs, labels) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(args.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(args.device) * args.nc_target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), args.nc_use_norm)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()


    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= args.nc_atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(args)
        print(" Updated !!!")

    # Show information
    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
        )
    )

    # Check early stop
    if args.nc_early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= args.nc_early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= args.nc_early_stop_patience
        ):
            print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= args.nc_atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= args.nc_patience:
                recorder.reset_state(args)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= args.nc_atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= args.nc_patience:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= args.nc_patience:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag

def outlier_detection(l1_norm_list, idx_mapping, args):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    if args.output_path is not None:
        # result_path = os.path.join(opt.result, opt.saving_prefix, opt.dataset)
        output_path = os.path.join(
            args.output_dir, "{}_{}_output.txt".format(args.attack_mode, args.data_set)
        )
        with open(args.output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )


def neural_cleanse(args):
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


    args.output_path = os.path.join(args.output_dir, "{}_{}_output_clean.txt".format(args.attack_mode, args.data_set))
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_path:
        with open(args.output_path, 'w+') as f:
            f.write("Output for cleanse:  - {}".format(args.attack_mode, args.data_set) + "\n")
    init_mask = np.ones((1, args.input_size, args.input_size)).astype(np.float32)
    init_pattern = np.ones((args.input_channel, args.input_size, args.input_size)).astype(np.float32)

    for test in range(args.nc_n_times_test):
        print("Test {}:".format(test))
        if args.output_path:
            with open(args.output_path, "a+") as f:
                f.write("-" * 30 + "\n")
                f.write("Test {}:".format(str(test)) + "\n")

        masks = []
        idx_mapping = {}

        for target_label in range(args.nc_total_label):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            args.nc_target_label = target_label
            recorder, args = train(args, init_mask, init_pattern)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1

        l1_norm_list = torch.stack([torch.norm(m, p=args.nc_use_norm) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping, args)