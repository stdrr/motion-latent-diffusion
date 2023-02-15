import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule

# From COSKAD
import pickle
import sys
sys.path.append('/media/odin/stdrr/projects/anomaly_detection/code/COSKAD/clean_code/HRAD_lightning')
from utils.dataset import PoseDatasetMorais
from utils.dataset_utils import ae_trans_list


class PoseDatasetForDiffusion(PoseDatasetMorais):

    def __init__(self, condition_length, **kwargs) -> None:
        self.condition_length = condition_length
        super().__init__(**kwargs)
        assert self.condition_length < self.seg_len, "condition length should be smaller than segment length"
        self.n_frames = self.seg_len - self.condition_length


    def __getitem__(self, index):
        item = super().__getitem__(index)
        item_dict = dict()
        if self.condition_length > 0:
            item_dict['motion'] = torch.tensor(item[0][:,self.n_frames:]).permute(1,2,0).contiguous().flatten(start_dim=1) # shape T,V,C
            item_dict['motion_cond'] = torch.tensor(item[0][:,:self.n_frames]).permute(1,2,0).contiguous().flatten(start_dim=1)
        else:
            item_dict['motion'] = torch.tensor(item[0]).permute(1,2,0).contiguous().flatten(start_dim=1) # shape T,V,C
            item_dict['motion_cond'] = item_dict['motion']
        item_dict['length'] = self.n_frames
        item_dict['coskad_input'] = item
        item_dict['text'] = '' # for compatibility
        # item_dict.update({f'metadata_{i}':m for i,m in enumerate(item[1:])})
        return item_dict
    

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
        self.cfg = args = cfg
        self.condition_length = cfg.DATASET.condition_len
        self.Dataset = PoseDatasetForDiffusion

        if self.cfg.DATASET.num_transform > 0:
            trans_list = ae_trans_list[:self.cfg.DATASET.num_transform]
        else: trans_list = None

        if args.DATASET.use_fitted_scaler:
            with open('{}/robust.pkl'.format(args.EXP_DIR), 'rb') as handle:
                scaler = pickle.load(handle)
            print('Scaler loaded from {}'.format('{}/robust.pkl'.format(args.EXP_DIR)))
        else: scaler = None

        self.dataset_args = {'path_to_morais_data': args.DATASET.UBNORMAL.ROOT, 'exp_dir': args.EXP_DIR,
                            'transform_list': trans_list, 'debug': args.DEBUG, 'headless': args.DATASET.headless,
                            'seg_len': args.DATASET.seg_len, 'normalize_pose': args.DATASET.normalize_pose, 'kp18_format': args.DATASET.kp18_format,
                            'vid_res': args.DATASET.vid_res, 'num_coords': args.DATASET.num_coords, 'sub_mean': args.DATASET.sub_mean,
                            'return_indices': False, 'return_metadata': True, 'return_mean': args.DATASET.sub_mean,
                            'symm_range': args.DATASET.symm_range, 'hip_center': args.DATASET.hip_center, 
                            'normalization_strategy': args.DATASET.normalization_strategy, 'ckpt': args.EXP_DIR, 'scaler': scaler, 
                            'kp_threshold':args.DATASET.kp_th, 'double_item': args.DATASET.double_item}

        if args.DEBUG:
            self._sample_set = self.get_sample_set(overrides=self.dataset_args)
            self._train_dataset = self._sample_set
        else:
            self._train_dataset = self.Dataset(condition_length=self.condition_length,
                                               include_global=False,
                                               split='train', **self.dataset_args)
            self._sample_set = self._train_dataset

        if args.VALIDATION and not args.DEBUG:
            self._val_dataset = self.Dataset(condition_length=self.condition_length,
                                            include_global=False,
                                            split='val', **self.dataset_args)
        elif args.VALIDATION:
            self._val_dataset = self.Dataset(condition_length=self.condition_length,
                                            include_global=False,
                                            split='val', **self.dataset_args)
        else:
            self._val_dataset = self._sample_set

        self.train_dataset = self._train_dataset
        self.val_dataset = self._val_dataset
        self.njoints = 18 if args.DATASET.kp18_format else 17
        self.nfeats = self.njoints * args.DATASET.num_coords
        
    def get_sample_set(self, overrides={}):
        self.dataset_args['debug'] = True
        return self.Dataset(self.condition_length, **self.dataset_args)

    def feats2joints(self, features):
        return features

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
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
