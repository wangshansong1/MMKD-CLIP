# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sklearn import metrics
import random
import torch
import json
import os
from tqdm import tqdm
import numpy as np
from torch import nn
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from scipy.special import softmax
import torch.nn.functional as F


def load_metadata(metadir="clipeval"):
    with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    return catalog


@torch.no_grad()
def build_text_features(args, templates, labels, model, tokenizer, skip_text_projection=False, classnorm=False):

    text_features = []
    if type(templates) == dict:
        class_similarities = []
        for cls_name, cls_text in templates.items():
            texts = tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True)
            class_embeddings = model.encode_text(texts)

            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            cls_sim = class_embeddings.mean(dim=0)  # equivalent to prompt ensembling

            class_similarities.append(cls_sim)

        text_features = torch.stack(class_similarities, dim=0)

    else:
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]

            texts = tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True)
            class_embeddings = model.encode_text(texts)

            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)
    mean, std = None, None

    if classnorm:
        mean, std = text_features.mean(dim=0)[None, :], text_features.std(dim=0)[None, :]
        text_features = (text_features - mean) / std

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, mean, std



@torch.no_grad()
def validate_zeroshot(args, val_loader, templates, labels, model, tokenizer, name, classnorm=False):
    # switch to evaluate mode

    model.cuda()
    model.eval()

    total_top1 = 0
    total_images = 0

    all_outputs = []
    all_targets = []

    text_features = None

    for samples in tqdm(val_loader):
        # Below if will run only for one iteration
        if text_features is None:
            print('=> encoding captions')
            if type(templates) == dict:
                for single_key in templates.keys():
                    length = len(templates[single_key])
                    templates[single_key] = templates[single_key][0:length]
                prompted_templates = templates
            else:
                prompted_templates = templates
            text_features, mean, std = build_text_features(args, prompted_templates, labels, model, tokenizer,
                                                               classnorm=classnorm)
        if isinstance(samples, tuple) or isinstance(samples, list):
            images, target = samples[0], samples[1]
        elif isinstance(samples, dict):
            images, target = samples["pixel_values"], samples["targets"]
        else:
            raise ValueError("unknown sample type", type(samples))

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # encode images
        image_features = model.encode_image(images)

        if classnorm:
            image_features = (image_features - mean) / std

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image.cpu()
        target = target.cpu()
        if name == "chexpert-5x200" or name == "DAD" or name == "chestmnist":
            # convert to label encoding
            target = torch.argmax(target, axis=1)

        # measure accuracy and record loss
        pred = logits_per_image.argmax(dim=1)
        correct = pred.eq(target).sum()
        total_top1 += correct.item()
        total_images += image_features.size(0)

        # Also save those to have results for the other metrics
        all_outputs.append(logits_per_image)
        all_targets.append(target)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = evaluate_metrics(all_outputs, all_targets)
    return {"acc": metrics['acc'], "acc_CIL": metrics['acc_CI'][0], "acc_CIH": metrics['acc_CI'][1],
            "auc_roc": metrics['auc_roc'], "auc_roc_CIL": metrics['auc_roc_CI'][0], "auc_roc_CIH": metrics['auc_roc_CI'][1],
            "f1_score": metrics['f1_score'], "f1_score_CIL": metrics['f1_score_CI'][0], "f1_score_CIH": metrics['f1_score_CI'][1],
            "precision_score": metrics['precision_score'], "precision_score_CIL": metrics['precision_score_CI'][0], "precision_score_CIH": metrics['precision_score_CI'][1],
            "recall_score": metrics['recall_score'], "recall_score_CIL": metrics['recall_score_CI'][0], "recall_score_CIH": metrics['recall_score_CI'][1],
            "acc_value": metrics["acc_value"],
            "auc_roc_value": metrics["auc_roc_value"],
            "f1_value": metrics["f1_value"],
            "precision_value": metrics["precision_value"],
            "recall_value": metrics["recall_value"],
            }

def evaluate_metrics(outputs, targets):
    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    n_classes = len(np.unique(targets))

    average_type = 'binary' if n_classes == 2 else 'macro'

    def roc_auc(outputs, targets):
        pos_score = outputs[:, 1] - outputs[:, 0]
        metric = metrics.roc_auc_score(targets, pos_score)

        return metric

    if outputs.ndim == 2:
        if not is_probabilities(outputs):
            outputs = softmax(outputs, axis=1)
        preds = np.argmax(outputs, axis=1)
        try:
            if average_type == 'binary':
                auc_val = roc_auc(outputs, targets)
            else:
                 auc_val = roc_auc_score(targets, outputs, multi_class='ovr', average='macro')
        except:
            auc_val = float("nan")
    elif outputs.ndim == 1:
        preds = (outputs >= 0.5).astype(int)
        try:
            auc_val = roc_auc_score(targets, outputs)
        except:
            auc_val = float("nan")
    else:
        raise ValueError(f"Unsupported output shape: {outputs.shape}")

    
    acc_val = accuracy_score(targets, preds)
    f1_val = f1_score(targets, preds, average=average_type)
    precision_val = precision_score(targets, preds, average=average_type)
    recall_val = recall_score(targets, preds, average=average_type)

    acc_ci = compute_bootstrap_ci(lambda o, t: accuracy_score(t, np.argmax(o, axis=1) if o.ndim == 2 else (o >= 0.5).astype(int)), outputs, targets)
    f1_ci = compute_bootstrap_ci(lambda o, t: f1_score(t, np.argmax(o, axis=1) if o.ndim == 2 else (o >= 0.5).astype(int), average=average_type), outputs, targets)
    precision_ci = compute_bootstrap_ci(lambda o, t: precision_score(t, np.argmax(o, axis=1) if o.ndim == 2 else (o >= 0.5).astype(int), average=average_type), outputs, targets)
    recall_ci = compute_bootstrap_ci(lambda o, t: recall_score(t, np.argmax(o, axis=1) if o.ndim == 2 else (o >= 0.5).astype(int), average=average_type), outputs, targets)
    auc_ci = compute_bootstrap_ci_auc(lambda o, t: roc_auc_score(t, o) if average_type == 'binary' else roc_auc_score(t, o, multi_class='ovr', average='macro'), outputs, targets)

    return {
        "acc": 100 * acc_val,
        "acc_CI": [100 * acc_ci[0], 100 * acc_ci[1]],
        "auc_roc": 100 * auc_val,
        "auc_roc_CI": [100 * auc_ci[0], 100 * auc_ci[1]],
        "f1_score": 100 * f1_val,
        "f1_score_CI": [100 * f1_ci[0], 100 * f1_ci[1]],
        "precision_score": 100 * precision_val,
        "precision_score_CI": [100 * precision_ci[0], 100 * precision_ci[1]],
        "recall_score": 100 * recall_val,
        "recall_score_CI": [100 * recall_ci[0], 100 * recall_ci[1]],

        "acc_value": acc_ci[2].tolist(),
        "auc_roc_value": auc_ci[2].tolist(),
        "f1_value": f1_ci[2].tolist(),
        "precision_value": precision_ci[2].tolist(),
        "recall_value": recall_ci[2].tolist(),
    }

def compute_bootstrap_ci(metric_fn, outputs, targets, n_iter=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    N = len(targets)
    n_classes = len(np.unique(targets))
    average_type = 'binary' if n_classes == 2 else 'macro'

    
    for _ in range(n_iter):
        idxs = stratified_bootstrap_indices(targets, N, rng)
        sample_outputs = outputs[idxs] if outputs.ndim == 1 else outputs[idxs, :]
        sample_targets = targets[idxs]
        try:
            value = metric_fn(sample_outputs, sample_targets)
            values.append(value)
        except:
            continue

    values = np.array(values)
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper, values

def compute_bootstrap_ci_auc(metric_fn, outputs, targets, n_iter=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    N = len(targets)
    n_classes = len(np.unique(targets))
    average_type = 'binary' if n_classes == 2 else 'macro'

    def roc_auc(outputs, targets):
        pos_score = outputs[:, 1] - outputs[:, 0]
        metric = metrics.roc_auc_score(targets, pos_score)

        return metric
    
    for _ in range(n_iter):
        idxs = stratified_bootstrap_indices(targets, N, rng)
        sample_outputs = outputs[idxs] if outputs.ndim == 1 else outputs[idxs, :]
        sample_targets = targets[idxs]
        try:
            if average_type == 'binary':
                value = roc_auc(sample_outputs, sample_targets)
            else:
                value = metric_fn(sample_outputs, sample_targets)
            values.append(value)
        except:
            continue

    values = np.array(values)
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper, values

def stratified_bootstrap_indices(targets, N, rng):
    """Ensure at least one sample from each class in each bootstrap."""
    classes = np.unique(targets)
    idxs = []

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        sampled = rng.choice(cls_indices)
        idxs.append(sampled)

    # Fill the rest randomly
    remaining = rng.integers(0, len(targets), size=N - len(classes))
    idxs.extend(remaining)

    return np.array(idxs)

def is_probabilities(x, tol=1e-3):
    x = np.asarray(x)
    if x.ndim == 2:
        row_sums = x.sum(axis=1)
        return np.all(np.abs(row_sums - 1.0) < tol)
    return False


