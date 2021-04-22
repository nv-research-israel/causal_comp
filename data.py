# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

from typing import NamedTuple

import numpy as np

from pathlib import Path
from ATTOP.data.dataset import sample_negative as ATTOP_sample_negative

import torch
from torch.utils import data

from useful_utils import categorical_histogram, get_and_update_num_calls
from COSMO_utils import temporary_random_numpy_seed


class DataItem(NamedTuple):
    """ A NamedTuple for returning a Dataset item """
    feat: torch.Tensor
    pos_attr_id: int
    pos_obj_id: int
    neg_attr_id: int
    neg_obj_id: int
    image_fname: str


class CompDataFromDict():
    # noinspection PyMissingConstructor
    def __init__(self, dict_data: dict, data_subset: str, data_dir: str):

        # define instance variables to be retrieved from struct_data_dict
        self.split: str = 'TBD'
        self.phase: str = 'TBD'
        self.feat_dim: int = -1
        self.objs: list = []
        self.attrs: list = []
        self.attr2idx: dict = {}
        self.obj2idx: dict = {}
        self.pair2idx: dict = {}
        self.seen_pairs: list = []
        self.all_open_pairs: list = []
        self.closed_unseen_pairs: list = []
        self.unseen_closed_val_pairs: list = []
        self.unseen_closed_test_pairs: list = []
        self.train_data: tuple = tuple()
        self.val_data: tuple = tuple()
        self.test_data: tuple = tuple()

        self.data_dir: str = data_dir

        # retrieve instance variables from struct_data_dict
        vars(self).update(dict_data)
        self.data = dict_data[data_subset]

        self.activations = {}
        features_dict = torch.load(Path(data_dir) / 'features.t7')
        for i, img_filename in enumerate(features_dict['files']):
            self.activations[img_filename] = features_dict['features'][i]

        self.input_shape = (self.feat_dim,)
        self.num_objs = len(self.objs)
        self.num_attrs = len(self.attrs)
        self.num_seen_pairs = len(self.seen_pairs)
        self.shape_obj_attr = (self.num_objs, self.num_attrs)

        self.flattened_seen_pairs_mask = self.get_flattened_pairs_mask(self.seen_pairs)
        self.flattened_closed_unseen_pairs_mask = self.get_flattened_pairs_mask(self.closed_unseen_pairs)
        self.flattened_all_open_pairs_mask = self.get_flattened_pairs_mask(self.all_open_pairs)
        self.seen_pairs_joint_class_ids = np.where(self.flattened_seen_pairs_mask)

        self.y1_freqs, self.y2_freqs, self.pairs_freqs = self._calc_freqs()
        self._just_load_labels = False

        self.train_pairs = self.seen_pairs

    def sample_negative(self, attr, obj):
        return ATTOP_sample_negative(self, attr, obj)

    def get_flattened_pairs_mask(self, pairs):
        pairs_ids = np.array([(self.obj2idx[obj], self.attr2idx[attr]) for attr, obj in pairs])
        flattened_pairs = np.zeros(self.shape_obj_attr, dtype=bool)  # init an array of False
        flattened_pairs[tuple(zip(*pairs_ids))] = True
        flattened_pairs = flattened_pairs.flatten()
        return flattened_pairs

    def just_load_labels(self, just_load_labels=True):
        self._just_load_labels = just_load_labels

    def get_all_labels(self):
        attrs = []
        objs = []
        joints = []
        self.just_load_labels(True)
        for attrs_batch, objs_batch in self:
            if isinstance(attrs_batch, torch.Tensor):
                attrs_batch = attrs_batch.cpu().numpy()
            if isinstance(objs_batch, torch.Tensor):
                objs_batch = objs_batch.cpu().numpy()
            joint = self.to_joint_label(objs_batch, attrs_batch)

            attrs.append(attrs_batch)
            objs.append(objs_batch)
            joints.append(joint)

        self.just_load_labels(False)
        attrs = np.array(attrs)
        objs = np.array(objs)
        return attrs, objs, joints

    def _calc_freqs(self):
        y2_train, y1_train, ys_joint_train = self.get_all_labels()
        y1_freqs = categorical_histogram(y1_train, range(self.num_objs), plot=False, frac=True)
        y1_freqs[y1_freqs == 0] = np.nan
        y2_freqs = categorical_histogram(y2_train, range(self.num_attrs), plot=False, frac=True)
        y2_freqs[y2_freqs == 0] = np.nan

        pairs_freqs = categorical_histogram(ys_joint_train,
                                            range(self.num_objs * self.num_attrs),
                                            plot=False, frac=True)
        pairs_freqs[pairs_freqs == 0] = np.nan
        return y1_freqs, y2_freqs, pairs_freqs

    def get(self, name):
        return vars(self).get(name)

    def __getitem__(self, idx):
        image_fname, attr, obj = self.data[idx]
        pos_attr_id, pos_obj_id = self.attr2idx[attr], self.obj2idx[obj]
        if self._just_load_labels:
            return pos_attr_id, pos_obj_id

        num_calls_cnt = get_and_update_num_calls(self.__getitem__)

        negative_attr_id, negative_obj_id = -1, -1  # default values
        if self.phase == 'train':
            # we set a temp np seed to override a weird issue with
            # sample_negative() at __getitem__, where the sampled pairs
            # could not be deterministically reproduced:
            # Now at each call to _getitem_ we set the seed to a 834276 (chosen randomly) + the number of calls to _getitem_
            with temporary_random_numpy_seed(834276 + num_calls_cnt):
                # draw a negative pair
                negative_attr_id, negative_obj_id = self.sample_negative(attr, obj)

        item = DataItem(
            feat=self.activations[image_fname],
            pos_attr_id=pos_attr_id,
            pos_obj_id=pos_obj_id,
            neg_attr_id=negative_attr_id,
            neg_obj_id=negative_obj_id,
            image_fname=image_fname,
        )
        return item

    def __len__(self):
        return len(self.data)

    def to_joint_label(self, y1_batch, y2_batch):
        return (y1_batch * self.num_attrs + y2_batch)


def get_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size,
                     num_workers=10, test_batchsize=None, shuffle_eval_set=True):
    if test_batchsize is None:
        test_batchsize = batch_size

    pin_memory = True
    if num_workers == 0:
        pin_memory = False
    print('num_workers = ', num_workers)
    print('pin_memory = ', pin_memory)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   pin_memory=pin_memory)
    valid_loader = None
    if valid_dataset is not None and len(valid_dataset) > 0:
        valid_loader = data.DataLoader(valid_dataset, batch_size=test_batchsize, shuffle=shuffle_eval_set,
                                       num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batchsize, shuffle=shuffle_eval_set,
                                  num_workers=num_workers, pin_memory=pin_memory)
    return test_loader, train_loader, valid_loader
