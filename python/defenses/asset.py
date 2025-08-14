import os
import argparse
import math
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize,RandomHorizontalFlip,RandomCrop
from tqdm.notebook import tqdm
# import torchshow as ts
import matplotlib.pyplot as plt
from poisoned_datasets import *
from models.resnet import ResNet18
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, Subset

from sklearn.mixture import GaussianMixture
def get_t(data, eps=1e-3):
    halfpoint = np.quantile(data, 0.5, interpolation='lower')
    lowerdata = np.array(data)[np.where(data<=halfpoint)[0]]
    f = np.ravel(lowerdata).astype(np.float32)
    f = f.reshape(-1,1)
    g = GaussianMixture(n_components=1,covariance_type='full')
    g.fit(f)
    weights = g.weights_
    means = g.means_ 
    covars = np.sqrt(g.covariances_)
    return (covars*np.sqrt(-2*np.log(eps)*covars*np.sqrt(2*np.pi)) + means)/ weights

def get_result(model, dataset, poi_idx):
    poi_set = Subset(dataset, poi_idx)
    clean_idx = list(set(np.arange(len(dataset))) - set(poi_idx))
    clean_set = Subset(dataset,clean_idx)
    
    poiloader = torch.utils.data.DataLoader(poi_set, batch_size=512, shuffle=False, num_workers=4)
    cleanloader = torch.utils.data.DataLoader(clean_set, batch_size=512, shuffle=False, num_workers=4)
    full_ce = nn.CrossEntropyLoss(reduction='none')
    
    poi_res = []
    for i, (data, target,_, _, _) in enumerate(tqdm(poiloader)):
        data, target= data.cuda(), target.cuda()
        with torch.no_grad():
            poi_outputs = model(data)
            # poi_loss = torch.var(poi_outputs,dim=1)
            poi_loss = full_ce(poi_outputs, target)
            poi_res.extend(poi_loss.cpu().detach().numpy())
            
    clean_res = []
    model.eval()
    for i, (data, target,_, _, _) in enumerate(tqdm(cleanloader)):
        data, target= data.cuda(), target.cuda()
        with torch.no_grad():
            clean_outputs = model(data)
            # clean_loss = torch.var(clean_outputs,dim=1)
            clean_loss = full_ce(clean_outputs, target)
            clean_res.extend(clean_loss.cpu().detach().numpy())
            
    return poi_res, clean_res

import statsmodels.api
def adjusted_outlyingness(series):
    _ao = []

    med = torch.median(series)
    q1, q3 = torch.quantile(series, torch.tensor([0.25, 0.75]).cuda())
    mc = torch.tensor(statsmodels.api.stats.stattools.medcouple(series.cpu().detach().numpy())).cuda()
    iqr = q3 - q1

    if mc > 0:
        w1 = q1 - (1.5 * torch.e ** (-4 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (3 * mc) * iqr)
    else:
        w1 = q1 - (1.5 * torch.e ** (-3 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (4 * mc) * iqr)

    for s in series:
        if s > med:
            _ao.append((s - med) / (w2 - med))
        else:
            _ao.append((med - s) / (med - w1))

    return torch.tensor(_ao).cuda()

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_size = 10, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)
    
    
def run_asset(args):
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
        
    args.mean, args.std = None, None
    device = args.device
    
    batch_size = 512
    nworkers = 8
    valid_size = 1000
    
    # val_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, download=False, transform=None)
    val_dataset, _ = build_backdoor_dataset(0, args, is_train=False)
    class my_subset(Dataset):
        r"""
        Subset of a dataset at specified indices.

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
            labels(sequence) : targets as required for the indices. will be the same length as indices
        """
        def __init__(self, dataset, indices,  transform):
            self.indices = indices
            if not isinstance(self.indices, list):
                self.indices = list(self.indices)
            self.dataset = Subset(dataset, self.indices)
            self.transform = transform

        def __getitem__(self, idx):
            image = self.dataset[idx][0]
            label = self.dataset[idx][1]
            if self.transform != None:
                # image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)
            return (image, label)

        def __len__(self):
            return len(self.indices)
        
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
        
    val_idx = []
    for i in range(10):
        current_label = np.where(np.array(val_dataset.dataset.targets)==i)[0]
        samples_idx = np.random.choice(current_label, size=int(valid_size/10), replace=False)
        val_idx.extend(samples_idx)

    val_set = Subset(val_dataset, val_idx)

    train_poi_set, _ = build_backdoor_dataset(args.attack_portion, args, selected_idx=poison_indices, is_train=True)
    train_poi_set.return_is_poison = True
    train_dataloader = torch.utils.data.DataLoader(train_poi_set, batch_size=batch_size, pin_memory=True, num_workers=8)
    o_poi_idx = list(train_poi_set.poisoned_indices)

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

    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['model'])
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    num_classes = 10
    
    

    def_model = ResNet18()
    def_model = def_model.to(device)
    optimizer = torch.optim.Adam(def_model.parameters(), lr=0.0001)
    optimizer2 = torch.optim.Adam(def_model.parameters(), lr=0.0001)

    criterion = nn.CrossEntropyLoss()

    full_ce = nn.CrossEntropyLoss(reduction='none')
    bce = torch.nn.MSELoss()

    for epoch in tqdm(range(1)):
        o_model2 = copy.deepcopy(model)
        o_model2.train()
        
        model_hat = copy.deepcopy(o_model2)
        layer_cake = list(model_hat.children())
        model_hat = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
        model_hat = model_hat.to(device)
        model_hat = model_hat.train()
        def_model.train()
        
        for iters, (input_train, target_train, _, _, poi) in enumerate(tqdm(train_dataloader)):
            pos_img,pos_lab,poi = input_train.cuda(), target_train.cuda(), poi.cuda()
            idxs = random.sample(range(valid_size), min(batch_size,valid_size))
            neg_img = torch.stack([val_set[i][0] for i in idxs]).to(device)
            neg_lab = torch.tensor([val_set[i][1] for i in idxs]).to(device)
            neg_outputs = def_model(neg_img)
            neg_loss = torch.mean(torch.var(neg_outputs,dim=1))
            optimizer.zero_grad()
            neg_loss.backward()
            optimizer.step()
            poi = poi.cuda()
                
            Vnet = MLP(input_size=8192, hidden_size=128, num_layers=2).to(device)
            Vnet.train()
            optimizer_hat = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
            optimizer_hat2 = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
            for _ in range(100):
                
                v_outputs = model_hat(pos_img)
                vneto = Vnet(v_outputs)
                v_label = torch.ones(v_outputs.shape[0]).to(device)
                rr_loss = bce(vneto.view(-1),v_label)
                Vnet.zero_grad()
                rr_loss.backward()
                optimizer_hat.step()
                
                vn_outputs = model_hat(neg_img)
                v_label2 = torch.zeros(vn_outputs.shape[0]).to(device)
                vneto2 = Vnet(vn_outputs)
                rr_loss2 = bce(vneto2.view(-1),v_label2)
                Vnet.zero_grad()
                rr_loss2.backward()
                optimizer_hat2.step()

            
            res = Vnet(v_outputs)
            pidx = torch.where(adjusted_outlyingness(res) > 2)[0]
            if pidx.shape[0] > 0:
                pos_outputs = def_model(pos_img[pidx])
                real_loss = -criterion(pos_outputs, pos_lab[pidx])
                optimizer2.zero_grad()
                real_loss.backward()
                optimizer2.step()
                print(neg_loss, real_loss)
            
        poi_res, clean_res = get_result(def_model, train_poi_set, o_poi_idx)
        
        poi_true = [1 for i in range(len(poi_res))]
        nor_true = [0 for i in range(len(clean_res))]

        true_label = poi_true + nor_true
        pred_label = poi_res + clean_res

        from sklearn.metrics import roc_auc_score, roc_curve, auc
        import matplotlib.pyplot as plt

        fpr, tpr, thersholds = roc_curve(true_label, pred_label)
        
        roc_auc = auc(fpr, tpr)
        print(roc_auc_score(true_label, pred_label))

        plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('asset.pdf')
        
        def plot_res(clean_res, poi_res):
            plt.figure(figsize=(3,1.5), dpi=300)
            plt.hist(np.array(clean_res), bins=200,label='Clean', color="#5da1f0")
            plt.hist(np.array(poi_res), bins=200,label='Poison', color="#f7d145")

            plt.ylabel("Number of samples")
            plt.xticks([])
            plt.ylim(0, 500)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
            plt.legend(prop={'size': 6})
            plt.show()
            
        
        total = poi_res + clean_res
        t = get_t(total, 1e-6)
        tp = len(o_poi_idx)-np.where(np.array(poi_res) < t)[0].shape[0]
        fp = len(clean_res)-np.where(np.array(clean_res) < t)[0].shape[0]
        fn = np.where(np.array(poi_res) < t)[0].shape[0]
        tn = np.where(np.array(clean_res) < t)[0].shape[0]
        print("tp:", tp)
        print("fp:", fp)
        print("fn:", fn)
        print("tpr:", tp/(tp+fn))
        print("fpr:", fp/(fp+tn))