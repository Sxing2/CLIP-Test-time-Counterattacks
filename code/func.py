import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).to(device)
std = torch.tensor(CIFAR100_STD).view(3, 1, 1).to(device)

def normalize(X):
    return (X - mu) / std
def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X

def rev_normalize(X):
    return X * std + mu
def reverse_clip_img_preprocessing(X):
    X = rev_normalize(X)
    return X

def multiGPU_CLIP_image_logits(images, model, text_tokens, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(None, None, None, model, image_tokens, text_tokens, prompt_token=prompt_token)[0]


def multiGPU_CLIP(args, model_image, model_text, model, images, text_tokens=None, prompt_token=None, dataset_name=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)
    if args is not None and dataset_name is not None:
        cache_prompts = os.path.join(args.cache, f"refined_{dataset_name.lower()}_prompts.pt")
        cache_wordnet_def = os.path.join(args.cache, f"refined_{dataset_name.lower()}_wn_def.pt")
    else:
        cache_prompts, cache_wordnet_def = None, None
    if cache_prompts is not None and os.path.exists(cache_prompts):
        text_features = torch.load(cache_prompts).to('cpu')
        if args.advanced_text == "wordnet_def":
            a_text_features = torch.load(cache_wordnet_def).to('cpu')
            text_features = (text_features + a_text_features) * 0.5
    else:
        text_features = model.module.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True) # [n_class, d_emb]
    text_features = text_features.to(device)
    image_features = model.module.encode_image(images, prompt_token)
    image_features = image_features / image_features.norm(dim=1, keepdim=True) # [bs, d_emb]
    logits_per_image = image_features @ text_features.t() * model.module.logit_scale.exp()
    logits_per_text = text_features @ image_features.t() * model.module.logit_scale.exp()

    return logits_per_image, logits_per_text, image_features, text_features

def kl_div(p_logits, q_logits):
    # p_logits, q_logits [bs, n_class] both have been softmax normalized
    kl_divs = (p_logits * (p_logits.log() - q_logits.log())).sum(dim=1) # [bs,]
    return kl_divs.mean()

def get_loss_general(tgt_logits, a_images, model_image_copy, text_features):
    # feed the perturbed image into the original visual encoder, regularise the predictive logits
    image_features = model_image_copy(a_images) # [bs, d_emb]
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logits_per_image_ = image_features @ text_features.t() * model_image_copy.module.logit_scale.exp() # [bs, n_class]
    l_general = kl_div(tgt_logits.softmax(dim=1), logits_per_image_.softmax(dim=1))
    # l_general = criterion_(F.log_softmax(logits_per_image_, dim=1), F.softmax(tgt_logits))
    return l_general

def get_loss_clean(clean_images, tgt_logits, model, text_features, prompt_token=None):
    # feed the clean image into the visual encoder, regularise the predictive logits
    image_features = model.module.encode_image(clean_images, prompt_token) # [bs, d_emb]
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits_per_image = image_features @ text_features.t() * model.module.logit_scale.exp() # [bs, n_class]
    l_clean = kl_div(tgt_logits.softmax(dim=1), logits_per_image.softmax(dim=1))
    # l_clean = criterion_(F.log_softmax(logits_per_image, dim=1), F.softmax(tgt_logits, dim=1))
    return l_clean