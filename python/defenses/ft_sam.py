import torch
import torch.nn as nn 
import tqdm 

import copy

# from sam import SAM

# from agem import *
from torch.utils.data import DataLoader
from poisoned_datasets import *
import sklearn
from torch.nn.modules.batchnorm import _BatchNorm
CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import contextlib
from torch.distributed import ReduceOp

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0    
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
    
        assert max_value >= min_value
        
        self.step() # take 1 step during initialization to get self._last_lr
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']
            
        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value

# class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, sam_alpha, rho_scheduler, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = sam_alpha
        
        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

# Training loop
def finetuning_sam(net, dataloader, optimizer, lr_scheduler, criterion, clean_loader, bd_loader, epochs, device, args):
    loss_hist = []
    try:
        print_and_log(args.logger, optimizer)
        print_and_log(args.logger, lr_scheduler)
    except:
        print(optimizer)
        print(lr_scheduler)

    for epoch in range(epochs):
        
        loss = finetune_epoch(net, dataloader, optimizer, lr_scheduler,
                            criterion, epoch, device, args)    
        clean_acc = test_model(net, clean_loader, device, args)
        poison_acc = test_model(net, bd_loader, device, args)
        
        loss_hist.append(loss)

        if epoch % 10 == 0:

            # print_and_log(args.logger, f'Fine-tuning epoch {epoch}: Loss: {loss}')
            print(f'SAM Fine-tuning epoch {epoch} Clean Accuracy: {clean_acc}')
            print(f'SAM Fine-tuning epoch {epoch} Poison Accuracy: {poison_acc}')

    return loss_hist


# Finetuning loop
def finetune_epoch(net, dataloader, optimizer, lr_scheduler, criterion, epoch, device, args):
    net.train()
    avg_loss = 0
    count = 0

    for inputs, targets, _, _ in dataloader:
        # if args.debug_mode and count > 2:
        #     break

        inputs, targets = inputs.to(device), targets.to(device)
        
        # first forward-backward step
        enable_running_stats(net)  # <- this is the important line
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(net)  # <- this is the important line
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        # def loss_fn(predictions, targets):
        #     return smooth_crossentropy(predictions, targets, smoothing=0.1).mean()
        # optimizer.set_closure(loss_fn, inputs, targets)
        # predictions, loss = optimizer.step()
        # lr_scheduler.step()
        # optimizer.update_rho_t()


        avg_loss += loss.item()
        count += 1
        # pbar.set_postfix({'loss': loss.item()})

    avg_loss = avg_loss/count

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)
        else:
            lr_scheduler.step()
        
    return avg_loss



def test_model(net, dataloader, device, args):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            


    accuracy = 100 * correct / total
    return accuracy

def FT_SAM(args):


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


    if args.model == 'myresnet18':
        from models.resnet import ResNet18, ResNet
        model = ResNet18(num_classes=args.nb_classes)
    elif args.model == 'mypreactresnet18':
        from models.preact_resnet import PreActResNet18, PreActResNet
        model = PreActResNet18(num_classes=args.nb_classes)
    elif args.model == 'mymnistnet':
        from models.mnist_net import MNISTNet
        model = MNISTNet()
    
    state_dict = torch.load(args.checkpoint)
    print("----------LOADING MODEL----------")
    model.load_state_dict(state_dict['model'])
    model.to(args.device)

    model.eval()

    args.mean, args.std = CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
    clean_dataset, _ = build_backdoor_dataset(0, args, is_train=False) # Attack portion should be 0 for for clean dataset
    clean_train_dataset, _ = build_backdoor_dataset(0, args, is_train=True)
    labels = torch.tensor(clean_train_dataset.targets)

    indices = sklearn.model_selection.train_test_split(labels, test_size=0.05, stratify=labels)[1]
    ft_dset = torch.utils.data.Subset(clean_train_dataset, indices)
    bd_dataset, _ = build_backdoor_dataset(1.0, args, is_train=False)

    finetune_loader = DataLoader(ft_dset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    clean_loader = DataLoader(clean_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    bd_loader = DataLoader(bd_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)



    finetune_optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.0005, momentum=0.9, rho=0.1)
    
    finetune_lr_scheduler = None

    print(finetune_optimizer)
    print(finetune_lr_scheduler)
    criterion = nn.CrossEntropyLoss()
    clean_acc = test_model(model, clean_loader, args.device, args)
    poison_acc = test_model(model, bd_loader, args.device, args)
    print(clean_acc, poison_acc)
    loss_hist_3 = finetuning_sam(model, finetune_loader, finetune_optimizer, finetune_lr_scheduler, 
                            criterion, clean_loader, bd_loader, epochs=50, device=args.device, 
                            args=args)
    
    clean_acc = test_model(model, clean_loader, args.device, args)
    poison_acc = test_model(model, bd_loader, args.device, args)
    print(clean_acc, poison_acc)
