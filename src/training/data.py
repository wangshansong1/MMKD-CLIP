import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

tokenizer = None

def set_tokenizer_value(context_length_from_args, encoder_name):
    tokenizer_kwargs = {}
    global tokenizer
    tokenizer = HFTokenizer(
        encoder_name,
        context_length=context_length_from_args,
        **tokenizer_kwargs,
    )
    
from open_clip import tokenize
from open_clip.tokenizer import HFTokenizer

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, teacherchoose, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')

        self.meta = [json.loads(line) for line in open(os.path.join(input_filename,teacherchoose+'.jsonl'))]
        self.transforms = transforms
        self.input_filename = input_filename
        
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        item = self.meta[idx]
        sample_id = item['id']
        folder_id = item['folderid']
        clip_names = item['available_clips']
        random_choice = random.choice(clip_names)

        clips_output = torch.load(os.path.join(self.input_filename, 'features', folder_id, sample_id, "clips_project_output.pt"))

        image_file = Image.open(
            os.path.join(self.input_filename,item['imgpath'])
        )
        images = self.transforms(image_file)
        texts = self.tokenize([item['txt']])[0]

        return images, texts

class KDCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, teacherchoose, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')

        self.meta = [json.loads(line) for line in open(os.path.join(input_filename,teacherchoose+'.jsonl'))]
        self.transforms = transforms
        self.input_filename = input_filename
        
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        item = self.meta[idx]
        sample_id = item['id']
        folder_id = item['folderid']
        clip_names = item['available_clips']
        random_choice = random.choice(clip_names)

        clips_output = torch.load(os.path.join(self.input_filename, 'features', folder_id, sample_id, "clips_project_output.pt"))

        image_file = Image.open(
            os.path.join(self.input_filename,item['imgpath'])
        )
        images = self.transforms(image_file)
        texts = self.tokenize([item['txt']])[0]
        iffea = clips_output[random_choice+'_img']
        tffea = clips_output[random_choice+'_txt']

        
        return images, texts, iffea, tffea

    

class MultiTaskDataLoader(object):
    """
    Multi-task DataLoader, the first dataloader is master dataloader
    """

    def __init__(self,
                 loaders, seed=0):
        assert len(loaders) > 1, "Less than 2 loader!"
        self.loaders = loaders
        self.iters = [iter(loader) for loader in loaders]
        self.lens = [len(loader) for loader in loaders]
        self.global_idx_in_cycle = 0
        self.seed = seed

    def __iter__(self):
        if self.global_idx_in_cycle > 0:
            self.iters[0] = iter(self.loaders[0])
        return self

    def __next__(self):
        output_tuple = (*next(self.iters[0]),)
        for k, (loader, _iter) in enumerate(zip(self.loaders[1:], self.iters[1:])):
            try:
                output_tuple += (*next(_iter),)
            except StopIteration:
                try:
                    loader.batch_sampler.sampler.set_epoch(int(self.global_idx_in_cycle // self.lens[k + 1]))
                except:
                    pass
                _iter = iter(loader)
                self.iters[k + 1] = _iter
                output_tuple += (*next(_iter),)

        if self.global_idx_in_cycle < sys.maxsize - 1:
            self.global_idx_in_cycle += 1
        else:
            self.global_idx_in_cycle = 0
        return output_tuple

    def __len__(self):
        return self.lens[0]


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards



def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data

    if args.dataset_type == 'pretrain':
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            teacherchoose=args.teacherchoose,
            sep=args.csv_separator,
            tokenizer=tokenizer)
    else:
        dataset = KDCsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            teacherchoose=args.teacherchoose,
            sep=args.csv_separator,
            tokenizer=tokenizer)

    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    if args.dataset_type == 'pretrain':
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True
        )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    set_tokenizer_value(args.tokenizer_context_length,args.text_encoder_model_name)
    if args.train_data:
        data["train"] = get_csv_dataset(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
   
    return data
