# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import json
import os
from constants import LC25000_lung, LC25000_colon
from clipeval import datasets, eval_zeroshot

def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)

def pil_collate_fn(batch):
    img = []
    label = []
    for i in batch:
        img.append(i[0])
        label.append(i[1])
    try:
        label = torch.tensor(label, dtype=torch.long)
    except:
        label = torch.vstack(label)
    return img, label

def r_collate_fn(batch):
    img = []
    label = []
    for i in batch:
        img.append(i[0])
        label.append(i[1])

    return img, label

@torch.no_grad()
def slip_evaluate(args, model, val_transform, tokenizer):
    
    metrics = {}
    if not is_master(args):
        return metrics
    
    catalog = eval_zeroshot.load_metadata("src/clipeval")

    if hasattr(model, "module"):
        model = model.module

    for d in catalog:
        
        val_dataset = datasets.get_downstream_dataset(
            args, catalog, d, is_train=False, transform=val_transform)
        
        if d == 'LC25000_lung':
            templates = LC25000_lung
            labels = None
        elif d == 'LC25000_colon':
            templates = LC25000_colon
            labels = None


        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size // 2, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False)
            

        metric = eval_zeroshot.validate_zeroshot(args, val_loader, templates, labels, model, tokenizer, d, classnorm=False)
        metrics[d] = metric
        json_str = json.dumps({"task": d, "metric": metric})
        
        if args.rank == 0:
            if os.path.isdir(args.output_dir):
                with open(os.path.join(args.output_dir, 'temp.txt'), mode="a+", encoding="utf-8") as f:
                    f.write(json_str + "\n")
            else:
                with open(os.path.join(args.output_dir), mode="a+", encoding="utf-8") as f:
                    f.write(json_str + "\n")

    return metrics
