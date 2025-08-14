import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct
from poisoned_datasets import *

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        x = self.out_conv(x)
        return x


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.block1 = nn.Sequential(
            ConvBrunch(3, 32, 3),
            ConvBrunch(32, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            ConvBrunch(32, 64, 3),
            ConvBrunch(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            ConvBrunch(64, 128, 3),
            ConvBrunch(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(4*4*128, 2)
        self.fc_size = 4*4*128

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def convert_dct2(data):
    for i in range(data.shape[0]):
        for c in range(3):
            data[i][:, :, c] = dct2(data[i][:, :, c])
    return data


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = 3 if args.data_set != 'MNIST' else 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.copy()
        for channel in range(self.n_channels):
            x_clone[channel, :, :] = x[channel, :, :] * self.variance[channel] + self.expected_values[channel]
        return x_clone

class FrequencyAnalysis():
    def __init__(self, args, input_size=32):
        if input_size == 32:
            self.clf = Detector().to(device)
            ckpt = torch.load(args.freq_detector_checkpoint) # trained on gtsrb, use for CIFAR evaluation 
            self.clf.load_state_dict(ckpt)
            self.clf.eval()
            self.denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)

    def train(self, data):
        # Already Pretrained
        return

    def predict(self, data):
        """
            data (np.array) b,h,w,c
        """
        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(data))):
                x, _, _, _ = data[i]
                x = x.clone().numpy()
                x = self.denormalizer(x)
                for c in range(3):
                    x[c, :, :] = dct2((x[c, :, :]*255).astype(np.uint8))
                x = torch.from_numpy(x)
                out = self.clf(x.unsqueeze(0).to(device)).detach().cpu()
                _, p = torch.max(out.data, 1)
                predictions.append(p)
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        return predictions

    def analysis(self, data):
        """
            data (np.array) b,h,w,c
        """
        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(data))):
                x, _, _, _ = data[i]
                x = x.clone().numpy()
                for c in range(3):
                    x[c, :, :] = dct2((x[c, :, :]*255).astype(np.uint8))
                x = torch.from_numpy(x)
                p = self.clf(x.unsqueeze(0).to(device)).detach().cpu()
                p = F.softmax(p, dim=1)
                predictions.append(p)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions[:, 1].detach().cpu().numpy()
        return predictions
    

def frequency_detect(args):
    detector = FrequencyAnalysis(args, args.input_size)

    clean_dataset, _ = build_eval_dataset(0, args)
    bd_dataset, _ = build_eval_dataset(1.0, args)

    

    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )
    bd_loader = torch.utils.data.DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    # 0 for clean, 1 for BD
    clean_y_pred = []
    bd_y_pred = []

    # for idx, (clean, bd) in enumerate(zip(clean_loader, bd_loader)):
    #     clean_img, _, _, _ = clean
    #     bd_img, _, _, _ = bd

    #     clean_img = clean_img.to(args.device)
    #     bd_img = bd_img.to(args.device)

    #     with torch.no_grad():
    clean_y_pred = detector.predict(clean_dataset)
    bd_y_pred = detector.predict(bd_dataset)

    clean_acc = np.sum((clean_y_pred == 0).astype(int)) / len(clean_y_pred)
    bd_detection_rate = np.sum((bd_y_pred == 1).astype(int)) / len(bd_y_pred)

    print(f'Clean Detection Rate: {clean_acc:.4f}')
    print(f'Backdoor Detection Rate: {bd_detection_rate:.4f}')

    tnr = np.sum((clean_y_pred == 0).astype(int)) / len(clean_y_pred)
    tpr = np.sum((bd_y_pred == 1).astype(int)) / len(bd_y_pred)
    fnr = np.sum((bd_y_pred == 0).astype(int)) / len(clean_y_pred)
    fpr = np.sum((clean_y_pred == 1).astype(int)) / len(clean_y_pred)

    print(f'TNR: {tnr:.4f}')
    print(f'TPR: {tpr:.4f}')
    print(f'FNR: {fnr:.4f}')
    print(f'FPR: {fpr:.4f}')


