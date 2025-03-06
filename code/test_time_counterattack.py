from __future__ import print_function

import argparse
import os
import time
import random
import logging
from tqdm import tqdm
from copy import deepcopy as dcopy

import torch
from torch.cuda.amp import GradScaler, autocast

import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import *
from attacks import *
from func import clip_img_preprocessing, multiGPU_CLIP

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--evaluate', type=bool, default=True) # eval mode
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--cache', type=str, default='./cache')    

    # test setting
    parser.add_argument('--test_set', default=[], type=str, nargs='*') # defaults to 17 datasets, if not specified
    parser.add_argument('--test_attack_type', type=str, default="pgd", choices=['pgd', 'CW', 'autoattack',])
    parser.add_argument('--test_eps', type=float, default=1,help='test attack budget')
    parser.add_argument('--test_numsteps', type=int, default=10)
    parser.add_argument('--test_stepsize', type=int, default=1)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['null_patch'], help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0, help='size for additional visual prompts')

    # data
    parser.add_argument('--root', type=str, 
                        default='./data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='tinyImageNet', 
                        help='dataset used for AFT methods')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    
    # TTC config
    parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
    parser.add_argument('--victim_resume', type=str, default=None, help='model weights of victim to attack.')
    parser.add_argument('--outdir', type=str, default=None, help='output directory for results')
    parser.add_argument('--tau_thres', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=2.,)
    parser.add_argument('--ttc_eps', type=float, default=4.)
    parser.add_argument('--ttc_numsteps', type=int, default=2)
    parser.add_argument('--ttc_stepsize', type=float, default=1.)

    args = parser.parse_args()
    return args

def compute_tau(clip_visual, images, n):
    orig_feat = clip_visual(clip_img_preprocessing(images), None) # [bs, 512]
    noisy_feat = clip_visual(clip_img_preprocessing(images + n), None)
    diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1) # [bs]
    return diff_ratio

def tau_thres_weighted_counterattacks(model, X, prompter, add_prompter, alpha, attack_iters, 
                           norm="l_inf", epsilon=0, visual_model_orig=None,
                           tau_thres:float=None, beta:float=None, clip_visual=None):
    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters == 0: # apply random noise (RN)
        return delta.data

    diff_ratio = compute_tau(clip_visual, X, delta.data) if clip_visual is not None else None

    # Freeze model parameters temporarily. Not necessary but for completeness of code
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
        )
        X_ori_norm = torch.norm(X_ori_reps, dim=-1) # [ bs]

    deltas_per_step = []
    deltas_per_step.append(delta.data.clone())

    for _step_id in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        if _step_id == 0 and diff_ratio is None: # compute tau at the zero-th step
            feature_diff = X_att_reps - X_ori_reps # [bs, 512]
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm # [bs]
        
        scheme_sign = (tau_thres - diff_ratio).sign()
        
        l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1) # [bs, numsteps+1, C, W, H]
    
    # create weights across steps
    weights = torch.arange(attack_iters+1).unsqueeze(0).expand(X.size(0), -1).to(device) # [bs, numsteps+1]
    weights = torch.exp(
        scheme_sign.view(-1, 1) * weights * beta
    ) # [bs, numsteps+1]
    weights /= weights.sum(dim=1, keepdim=True)

    weights_hard = torch.zeros_like(weights) # [bs, numsteps+1]
    weights_hard[:,0] = 1.

    weights = torch.where(scheme_sign.unsqueeze(1)>0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters+1, 1, 1, 1)
    
    Delta = (weights * Delta).sum(dim=1)
    
    # Unfreeze model parameters. Only for completeness of code
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return Delta


def validate(args, val_dataset_name, model, model_text, model_image,
             prompter, add_prompter, criterion, visual_model_orig=None,
             clip_visual=None
    ):
    
    logging.info(f"Evaluate with Attack method: {args.test_attack_type}")

    dataset_num = len(val_dataset_name)
    all_clean_org, all_clean_ttc, all_adv_org, all_adv_ttc = {},{},{},{}

    test_stepsize = args.test_stepsize

    ttc_eps = args.ttc_eps
    ttc_numsteps = args.ttc_numsteps
    ttc_stepsize = args.ttc_stepsize
    beta = args.beta
    tau_thres = args.tau_thres

    for cnt in range(dataset_num):
        val_dataset, val_loader = load_val_dataset(args, val_dataset_name[cnt])
        dataset_name = val_dataset_name[cnt]
        texts = get_text_prompts_val([val_dataset], [dataset_name])[0]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_org_ttc = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_ttc = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        text_tokens = clip.tokenize(texts).to(device)
        end = time.time()
        
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)

            with autocast():

                # original acc of clean images
                with torch.no_grad():
                    clean_output,_,_,_ = multiGPU_CLIP(
                        None, None, None, model, prompter(clip_img_preprocessing(images)),
                        text_tokens = text_tokens,
                        prompt_token = None, dataset_name = dataset_name
                    )
                    clean_acc = accuracy(clean_output, target, topk=(1,))
                    top1_org.update(clean_acc[0].item(), images.size(0))

                # TTC on clean images
                ttc_delta_clean = tau_thres_weighted_counterattacks(
                    model, images, prompter, add_prompter,
                    alpha=ttc_stepsize, attack_iters=ttc_numsteps,
                    norm='l_inf', epsilon=ttc_eps, visual_model_orig=None,
                    tau_thres=tau_thres, beta = beta,
                    clip_visual=clip_visual
                )
                with torch.no_grad():
                    clean_output_ttc,_,_,_ = multiGPU_CLIP(
                        None, None, None, model, prompter(clip_img_preprocessing(images+ttc_delta_clean)),
                        text_tokens = text_tokens,
                        prompt_token = None, dataset_name = dataset_name
                    )
                    clean_acc_ttc = accuracy(clean_output_ttc, target, topk=(1,))
                    top1_org_ttc.update(clean_acc_ttc[0].item(), images.size(0))

                # generate adv samples for this batch
                torch.cuda.empty_cache()
                if args.test_attack_type == "pgd":
                    delta_prompt = attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion,
                                              images, target, test_stepsize, args.test_numsteps, 'l_inf',
                                              text_tokens=text_tokens, epsilon=args.test_eps, dataset_name=dataset_name)
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "CW":
                    delta_prompt = attack_CW(args, prompter, model, model_text, model_image, add_prompter, criterion,
                                             images, target, text_tokens,
                                             test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "autoattack":
                    attacked_images = attack_auto(model, images, target, text_tokens,
                        None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run)

                # acc of adv images without ttc
                with torch.no_grad():
                    adv_output,_,_,_ = multiGPU_CLIP(
                        None,None,None, model, prompter(clip_img_preprocessing(attacked_images)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name 
                    )
                    adv_acc = accuracy(adv_output, target, topk=(1,))
                    top1_adv.update(adv_acc[0].item(), images.size(0))
                
                ttc_delta_adv = tau_thres_weighted_counterattacks(
                    model, attacked_images.data, prompter, add_prompter,
                    alpha=ttc_stepsize, attack_iters=ttc_numsteps,
                    norm='l_inf', epsilon=ttc_eps, visual_model_orig=None,
                    tau_thres=tau_thres, beta = beta,
                    clip_visual = clip_visual
                )
                with torch.no_grad():
                    adv_output_ttc,_,_,_ = multiGPU_CLIP(
                        None,None,None, model, prompter(clip_img_preprocessing(attacked_images+ttc_delta_adv)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name 
                    )
                    adv_output_acc = accuracy(adv_output_ttc, target, topk=(1,))
                    top1_adv_ttc.update(adv_output_acc[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        
        torch.cuda.empty_cache()
        clean_acc = top1_org.avg
        clean_ttc_acc = top1_org_ttc.avg
        adv_acc = top1_adv.avg
        adv_ttc_acc = top1_adv_ttc.avg

        all_clean_org[dataset_name] = clean_acc
        all_clean_ttc[dataset_name] = clean_ttc_acc
        all_adv_org[dataset_name] = adv_acc
        all_adv_ttc[dataset_name] = adv_ttc_acc

        show_text = f"{dataset_name}:\n\t"
        show_text += f"- clean acc.  {clean_acc:.2f} (ttc: {clean_ttc_acc:.2f})\n\t"
        show_text += f"- robust acc. {adv_acc:.2f} (ttc: {adv_ttc_acc:.2f})"
        
        logging.info(show_text)

    all_clean_org_avg = np.mean([all_clean_org[name] for name in all_clean_org]).item()
    all_clean_ttc_avg = np.mean([all_clean_ttc[name] for name in all_clean_ttc]).item()
    all_adv_org_avg = np.mean([all_adv_org[name] for name in all_adv_org]).item()
    all_adv_ttc_avg = np.mean([all_adv_ttc[name] for name in all_adv_ttc]).item()
    show_text = f"===== SUMMARY ACROSS {dataset_num} DATASETS =====\n\t"
    show_text += f"AVG acc. {all_clean_org_avg:.2f} (ttc: {all_clean_ttc_avg:.2f})\n\t"
    show_text += f"AVG acc. {all_adv_org_avg:.2f} (ttc: {all_adv_ttc_avg:.2f})"
    logging.info(show_text)

    zs_clean_org_avg = np.mean([all_clean_org[name] for name in val_dataset_name if name != args.dataset]).item()
    zs_clean_ttc_avg = np.mean([all_clean_ttc[name] for name in val_dataset_name if name != args.dataset]).item()
    zs_adv_org_avg = np.mean([all_adv_org[name] for name in val_dataset_name if name != args.dataset]).item()
    zs_adv_ttc_avg = np.mean([all_adv_ttc[name] for name in val_dataset_name if name != args.dataset]).item()
    show_text = f"===== SUMMARY ACROSS {dataset_num-1} DATASETS (EXCEPT {args.dataset}) =====\n\t"
    show_text += f"AVG acc. {zs_clean_org_avg:.2f} (ttc: {zs_clean_ttc_avg:.2f})\n\t"
    show_text += f"AVG acc. {zs_adv_org_avg:.2f} (ttc: {zs_adv_ttc_avg:.2f})"
    logging.info(show_text)

    return all_clean_org_avg, all_clean_ttc_avg, all_adv_org_avg, all_adv_ttc_avg

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    args = parse_options()

    outdir = args.outdir if args.outdir is not None else "TTC_results"
    outdir = os.path.join(outdir, f"{args.test_attack_type}_eps_{args.test_eps}_numsteps_{args.test_numsteps}")
    os.makedirs(outdir, exist_ok=True)

    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_filename = ""
    log_filename += f"ttc_eps_{args.ttc_eps}_thres_{args.tau_thres}_beta_{args.beta}_numsteps_{args.ttc_numsteps}_stepsize_{int(args.ttc_stepsize)}_seed_{seed}.log".replace(" ", "")
    log_filename = os.path.join(outdir, log_filename)
    logging.basicConfig(
        filename = log_filename,
        level = logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(args)

    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.

    imagenet_root = '.data/ImageNet'
    tinyimagenet_root = "./data/tiny-imagenet-200"
    args.imagenet_root = imagenet_root
    args.tinyimagenet_root = tinyimagenet_root

    # load model
    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)

    if args.victim_resume: # employ TTC on AFT checkpoints
        clip_visual = dcopy(model.visual)
        model = load_checkpoints2(args, args.victim_resume, model, None)
    else:                  # employ TTC on the original CLIP
        clip_visual = None

    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()
    logging.info("done loading model.")

    if len(args.test_set) == 0:
        test_set = DATASETS
    else:
        test_set = args.test_set

    # criterion to compute attack loss, the reduction of 'sum' is important for effective attacks
    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    validate(
        args, test_set, model, None, None, prompter,
        add_prompter, criterion_attack, None, clip_visual
    )

if __name__ == "__main__":
    main()