# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the following repo
# (released under the MIT License).
#
# Source:
# https://github.com/Tushar-N/attributes-as-operators/blob/master/data/dataset.py
#
# The license for the original version of this file can be
# found in ATTOP/LICENSE
# ---------------------------------------------------------------

import numpy as np

def sample_negative(self, attr, obj):
    new_attr, new_obj = self.train_pairs[np.random.choice(len(self.train_pairs))]
    if new_attr == attr and new_obj == obj:
        return self.sample_negative(attr, obj)
    return (self.attr2idx[new_attr], self.obj2idx[new_obj])
