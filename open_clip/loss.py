# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False
import numpy as np

hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)

        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class KDClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, t_image_features, t_text_features):
        device = image_features.device

        # --- Step 1: Gather features if needed ---
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            t_all_image_features, t_all_text_features = gather_features(
                t_image_features, t_text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
        else:
            all_image_features, all_text_features = image_features, text_features
            t_all_image_features, t_all_text_features = t_image_features, t_text_features

        # --- Step 2: Normalize features (only raw features) ---
        normalized_image_features = F.normalize(image_features, dim=1)
        normalized_text_features = F.normalize(text_features, dim=1)
        normalized_all_image_features = F.normalize(all_image_features, dim=1)
        normalized_all_text_features = F.normalize(all_text_features, dim=1)
        # Note: t_image_features and t_text_features already normalized before input!
        # So no need to normalize t_all_image_features, t_all_text_features again!

        local_batch_size = image_features.shape[0]
        global_batch_size = normalized_all_text_features.shape[0]

        # --- Step 3: Compute logits ---
        if self.local_loss:
            logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
            logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
        else:
            logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
            logits_per_text = logits_per_image.T

        # --- Step 4: Prepare labels ---
        if self.world_size > 1:
            if self.local_loss:
                start_idx = self.rank * local_batch_size
                labels = torch.arange(start_idx, start_idx + local_batch_size, device=device, dtype=torch.long)
            else:
                labels = torch.arange(global_batch_size, device=device, dtype=torch.long)
        else:
            labels = torch.arange(local_batch_size, device=device, dtype=torch.long)

        # Always prepare global_labels for icl_loss
        global_labels = torch.arange(global_batch_size, device=device, dtype=torch.long)

        # Optionally cache labels
        if self.cache_labels:
            if device not in self.labels or self.prev_num_logits != global_batch_size:
                self.labels[device] = (labels, global_labels)
                self.prev_num_logits = global_batch_size
            else:
                labels, global_labels = self.labels[device]

        # --- Step 5: Feature Distillation Loss (fd_loss) ---
        fd_loss = (
            F.mse_loss(normalized_all_image_features, t_all_image_features) +
            F.mse_loss(normalized_all_text_features, t_all_text_features)
        )

        # --- Step 6: ICL Loss (icl_loss) ---
        logits_per_s_image_to_t_text = self.cross_logit_scale * normalized_all_image_features @ t_all_text_features.T
        logits_per_s_text_to_t_image = self.cross_logit_scale * normalized_all_text_features @ t_all_image_features.T

        icl_loss = (
            F.cross_entropy(logits_per_s_image_to_t_text, global_labels) +
            F.cross_entropy(logits_per_s_text_to_t_image, global_labels)
        ) / 2

        # --- Step 7: Task Loss (task_loss) ---
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return task_loss, fd_loss, icl_loss

