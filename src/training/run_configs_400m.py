# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from configs import Config

@dataclass
class b16_400m(Config):
    inmem=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    data_root = '/path/MMKDCLIP_DATA/'
    train_data="/path/MMKDCLIP_DATA/"
    workers=8
    batch_size=4
    epochs= 20
    eval_freq = 1
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    nodes=1
    ngpus=1
    dataset_type = "pretrain"
    precision = "fp32"
    imagenet_val = None
    report_to = 'wandb'
    lr=5e-5
    tokenizer_context_length = 256
    evalclipname = 'D9MCCLIP'
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True
    pretrained = ''
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

@dataclass
class b16_kd(Config):
    inmem=True
    engine="train_one_epoch_kd"
    eval_steps=5000
    save_frequency=1
    data_root = "/path/MMKDCLIP_DATA/"
    train_data="/path/MMKDCLIP_DATA/"
    workers=8
    batch_size=4
    epochs= 20
    eval_freq = 1
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    nodes=1
    ngpus=1
    precision = "amp"
    imagenet_val = None
    report_to = 'wandb'
    dataset_type = "kd"
    kdweight = 50.0
    lr=1e-6
    logs = ""
    tokenizer_context_length = 256
    pretrained = ''
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    evalclipname = 'D9MCCLIP'

@dataclass
class eval(Config):
    inmem=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data=""
    workers=8
    eval_freq = 1
    batch_size=512
    epochs=1
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    nodes=1
    ngpus=1
    precision = "fp32"
    imagenet_val = None
    data_root = 'Path/MMKDCLIP_DATA/'
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True
    pretrained = 'Path/model.pt'
    text_encoder_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    tokenizer_context_length = 256

if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)