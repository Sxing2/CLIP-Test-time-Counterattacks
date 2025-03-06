import shutil
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import *
from torch.utils.data import DataLoader
import json

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names

def save_checkpoint(state, save_folder, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(save_folder, filename)
    bestfile = os.path.join(save_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')

def assign_learning_rate(optimizer, new_lr, tgt_group_idx=None):
    for group_idx, param_group in enumerate(optimizer.param_groups):
        if tgt_group_idx is None or tgt_group_idx==group_idx:
            param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps, tgt_group_idx=None):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr, tgt_group_idx)
        return lr
    return _lr_adjuster

def null_scheduler(init_lr):
    return lambda step:init_lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    # print(dict_imagenet_folder2name)
    return dict_imagenet_folder2name



def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels.cpu()]


preprocess = transforms.Compose([
    transforms.ToTensor()
])
preprocess224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
preprocess224_caltech = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])
preprocess224_interpolate = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_train_dataset(args):
    if args.dataset == 'cifar100':
        return CIFAR100(args.root, transform=preprocess, download=True, train=True)
    elif args.dataset == 'cifar10':
        return CIFAR10(args.root, transform=preprocess, download=True, train=True)
    elif args.dataset == 'ImageNet':
        assert args.imagenet_root is not None
        print(f"Loading ImageNet from {args.imagenet_root}")
        return ImageFolder(os.path.join(args.imagenet_root, 'train'), transform=preprocess224)
    else:
        print(f"Train dataset {args.dataset} not implemented")
        raise NotImplementedError

def get_eval_files(dataset_name):
    # only for imaegnet and tinyimagenet
    refined_data_file = f"{dataset_name.lower()}_refined_labels.json"
    refined_data = read_json(refined_data_file)
    eval_select = {ssid:refined_data[ssid]['eval_files'] for ssid in refined_data}
    return eval_select

def load_val_dataset(args, val_dataset_name):

    if val_dataset_name == 'cifar10':
        val_dataset = CIFAR10(args.root, transform=preprocess224, download=True, train=False)

    elif val_dataset_name == 'cifar100':
        val_dataset = CIFAR100(args.root, transform=preprocess224, download=True, train=False)

    elif val_dataset_name == 'Caltech101':
        val_dataset = Caltech101(args.root, target_type='category', transform=preprocess224_caltech, download=True)

    elif val_dataset_name == 'PCAM':
        val_dataset = PCAM(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'STL10':
        val_dataset = STL10(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'SUN397':
        val_dataset = SUN397(args.root, transform=preprocess224, download=True)

    elif val_dataset_name == 'StanfordCars':
        val_dataset = StanfordCars(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'Food101':
        val_dataset = Food101(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'oxfordpet':
        val_dataset = OxfordIIITPet(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'EuroSAT':
        val_dataset = EuroSAT(args.root, transform=preprocess224, download=True)

    elif val_dataset_name == 'Caltech256':
        val_dataset = Caltech256(args.root, transform=preprocess224_caltech, download=True)

    elif val_dataset_name == 'flowers102':
        val_dataset = Flowers102(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'Country211':
        val_dataset = Country211(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'dtd':
        val_dataset = DTD(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'fgvc_aircraft':
        val_dataset = FGVCAircraft(args.root, split='test', transform=preprocess224, download=True)
    
    elif val_dataset_name == 'ImageNet':
        from torchvision.datasets import ImageNetFolder
        if args.evaluate:
            val_dataset = ImageNetFolder(os.path.join(args.imagenet_root, 'val'), transform=preprocess224)
        else:
            eval_select = get_eval_files(val_dataset_name)
            val_dataset = ImageNetFolder(os.path.join(args.imagenet_root, 'train'), transform=preprocess224, select_files=eval_select)
    
    elif val_dataset_name == 'tinyImageNet':
        from torchvision.datasets import ImageNetFolder
        if args.evaluate:
            val_dataset = ImageNetFolder(os.path.join(args.tinyimagenet_root, 'val_'), transform=preprocess224)
        else:
            eval_select = get_eval_files(val_dataset_name)
            val_dataset = ImageNetFolder(os.path.join(args.tinyimagenet_root, 'train'), transform=preprocess224, select_files=eval_select)
    else:
        print(f"Val dataset {val_dataset_name} not implemented")
        raise NotImplementedError
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, 
               num_workers=args.num_workers, shuffle=False,)
    
    return val_dataset, val_loader

def get_text_prompts_train(args, train_dataset, template='This is a photo of a {}'):
    class_names = train_dataset.classes
    if args.dataset == 'ImageNet':
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]
    return texts_train

def get_text_prompts_val(val_dataset_list, val_dataset_name, template='This is a photo of a {}.'):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes if hasattr(each, 'classes') else each.clip_categories
            if val_dataset_name[cnt] in ['ImageNet', 'tinyImageNet']:
                refined_data = read_json(f"{val_dataset_name[cnt].lower()}_refined_labels.json")
                clean_class_names = [refined_data[ssid]['clean_name'] for ssid in class_names]
                class_names = clean_class_names
                
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list


def read_json(json_file:str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# def get_prompts(class_names):
#     # consider using correct articles
#     template = "This is a photo of a {}."
#     template_v = "This is a photo of an {}."
#     prompts = []
#     for class_name in class_names:
#         if class_name[0].lower() in ['a','e','i','o','u'] or class_name == "hourglass":
#             prompts.append(template_v.format(class_name))
#         else:
#             prompts.append(template.format(class_name))
#     return prompts

def freeze(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad=False
    return

DATASETS = [
    'cifar10', 'cifar100', 'STL10','ImageNet',
    'Caltech101', 'Caltech256', 'oxfordpet', 'flowers102', 'fgvc_aircraft',
    'StanfordCars', 'SUN397', 'Country211', 'Food101', 'EuroSAT',
    'dtd', 'PCAM', 'tinyImageNet',
]


def write_file(txt:str, file:str, mode='a'):
    with open(file, mode) as f:
        f.write(txt)


def load_resume_file(file:str, gpu:int):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        if gpu is None:
            checkpoint = torch.load(file)
        else:
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(file, map_location=loc)
        print("=> loaded checkpoint '{}' (epoch {})".format(file, checkpoint['epoch']))
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return None 


def load_checkpoints2(args, resume_file, model, optimizer=None):
    checkpoint = load_resume_file(resume_file, None)
    try:
        model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
    except:
        model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model
