import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule

# From COSKAD
import argparse
import yaml
import sys
sys.path.append('/media/odin/stdrr/projects/anomaly_detection/code/COSKAD/clean_code/HRAD_lightning')
from utils.argparser import init_sub_args
from utils.dataset import get_dataset_and_loader


class UBnormal(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "ubnormal"
        args, dataset_args = cfg

        if args.validation:
            train_dataset, train_loader, val_dataset, val_loader = get_dataset_and_loader(dataset_args,split=args.split, validation=args.validation)
            self.Dataset_val = val_dataset.__class__
            self._val_dataset = val_dataset
            self._val_loader = val_loader
        else:
            train_dataset, train_loader = get_dataset_and_loader(dataset_args,split=args.split, validation=args.validation)
            self._val_dataset = train_dataset
            self._val_loader = train_loader
        self.Dataset = train_dataset.__class__
        self._train_dataset = train_dataset
        self._train_loader = train_loader
        self.cfg = args
        sample_overrides = {
            "split": "val" if args.validation else "train",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.njoints = self._sample_set.V
        self.nfeats = self.njoints * self._sample_set.num_coords
        # self.transforms = self._sample_set.transforms

    def get_sample_set(self, overrides={}):
        if overrides['split'] == 'val':
            return self._val_dataset
        return self._train_dataset

    def feats2joints(self, features):
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = features * std + mean
        # return recover_from_ric(features, self.njoints)
        return features

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list


    # override the methods
    def train_dataloader(self):
        return self._train_loader

    def predict_dataloader(self):
        raise NotImplementedError('Predict dataloader is not implemented')

    def val_dataloader(self):
        # overrides batch_size and num_workers
        return self._val_loader
