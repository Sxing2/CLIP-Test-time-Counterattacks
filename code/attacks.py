import torch
from utils import one_hot_embedding
# from models.model import *
import torch.nn.functional as F
import functools
from autoattack import AutoAttack
from func import clip_img_preprocessing, multiGPU_CLIP, multiGPU_CLIP_image_logits

lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_CW(args, prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
              attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
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
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
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
        delta.grad.zero_()

    return delta


def attack_CW_noprompt(args, prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
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
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, _images, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
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
        delta.grad.zero_()

    return delta

def attack_unlabelled(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    delta = torch.zeros_like(X)
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

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters <= 0: 
        return delta

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        # l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))**(0.5)).sum()
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

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta

#### opposite update direction of attack_unlabelled()
def attack_unlabelled_opp(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    delta = torch.zeros_like(X)
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

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        # l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))**(0.5)).sum()
        l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))).sum()

        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            # d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            # d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = (d - scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta

def attack_unlabelled_cosine(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    # unlabelled attack to maximise cosine similarity between the attacked image
    # and the original image, computed by PGD
    delta = torch.zeros_like(X)
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

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        # X_ori_reps_norm = X_ori_reps / X_ori_reps.norm(dim=-1, keepdim=True)

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token) # [bs, d_out]
        # X_att_reps_norm = X_att_reps / X_att_reps.norm(dim=-1, keepdim=True)
        
        cos_loss = 1 - F.cosine_similarity(X_att_reps, X_ori_reps) # [bs]

        grad = torch.autograd.grad(cos_loss, delta)[0]
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

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta


def attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion, X, target, alpha,
               attack_iters, norm, text_tokens=None, restarts=1, early_stop=True, epsilon=0, dataset_name=None):
    delta = torch.zeros_like(X).cuda()
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

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    for iter in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, prompted_images, 
                                  text_tokens=text_tokens, prompt_token=prompt_token, dataset_name=dataset_name)

        loss = criterion(output, target)

        grad = torch.autograd.grad(loss, delta)[0]

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
        # delta.grad.zero_()

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta


def attack_pgd_noprompt(args, prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
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
    for _ in range(attack_iters):

        _images = clip_img_preprocessing(X + delta)
        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, _images, text_tokens, None)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
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
        delta.grad.zero_()

    return delta

def attack_auto(model, images, target, text_tokens, prompter, add_prompter,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens,
        prompter=None, add_prompter=None
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv