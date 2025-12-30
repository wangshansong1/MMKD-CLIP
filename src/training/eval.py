# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import logging
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
from open_clip import create_model_and_transforms, trace_model, get_mean_std
import open_clip
from training.distributed import init_distributed_device
from training.params import parse_args
from training.slip_evaluate import slip_evaluate
from PIL import Image

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main(args=None):
    if args is None:
        args = parse_args()

    device = init_distributed_device(args)
    random_seed(args.seed, args.rank)

    args.model = args.model.replace('/', '-')
    mean, std = get_mean_std(args)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        mean=mean, std=std,
        inmem=hasattr(args, "inmem"),
        clip_model=args.clip_model,
        text_encoder_name=args.text_encoder_model_name,
    )
    
    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('_orig_mod'):
                    sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
            else:
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    context_length = args.tokenizer_context_length
    tokenizer_kwargs = {}
    tokenize = open_clip.HFTokenizer(
        args.text_encoder_model_name,
        context_length=context_length,
        **tokenizer_kwargs,
    )

    slip_evaluate(args, model, preprocess_val, tokenize)


if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from configs import search_config
     
    config = search_config('eval')
    
    config.resume = f"path/model.pt"
    config.output_dir = f'path/result.txt'
    main(config)