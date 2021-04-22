# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

import os
import sys
from collections import namedtuple
from pathlib import Path
import numpy as np

from dataclasses import dataclass
from simple_parsing import ArgumentParser, ConflictResolution, field

from useful_utils import to_simple_parsing_args


@dataclass
class DataCfg():
    # dataset name
    dataset_name: str = 'ao_clevr'

    # default: {dataset_basedir}/{dataset_name}
    data_dir: str = 'default'

    dataset_basedir: str = 'data'

    # VT_random|UV_random
    dataset_variant: str = 'VT_random'

    num_split: int = 5000

    metadata_from_pkl: bool = True

    def __post_init__(self):
        if self.data_dir == 'default':
            self.data_dir = os.path.join(self.dataset_basedir, self.dataset_name)

    def __getitem__(self, key):
        """ Allow accessing instance attributes as dictionary keys """
        return getattr(self, key)


@dataclass
class MetricsCfg:
    calc_AUC: bool = True
    """ A flag to calc AUC metric (slower) """

    def __post_init__(self):
        assert self.calc_AUC


@dataclass(frozen=True)
class EarlyStopMetric:
    # metric name
    metric: str

    # min: minimize or max: maximize
    polarity: str



@dataclass
class ModelCfg():
    """" The hyper params to configure the architecture.
         Note: For backward compatibility, some of the variable names here are different than those mentioned in the
         paper. We explain in comments, how each variable is referenced at the paper.
    """

    h_dim: int = 150
    """ Number of hidden units in each MLP layer. """

    nlayers_label: int = 2
    """ Number of MLP layer, for h_O anb h_A. """

    nlayers_joint_ao: int = 2
    """ Number of MLP layers, for g. """


    E_num_hidden: int = 2
    """ Number of layers, for g^-1_A and g^-1_O """

    E_num_common_hidden: int = 0
    """ Number of layers, when projecting pretrained image features to the feature space \X 
    (as explained for Zappos in section C.1)
    """

    mlp_activation: str = 'leaky_relu'

    learn_embeddings: bool = True
    """ Set to False to train as VisProd, with CE loss rather than an embedding loss. """

    def __post_init__(self):
        self.VisProd: bool = not self.learn_embeddings

        assert self.E_num_common_hidden <= self.E_num_hidden


@dataclass
class TrainCfg:
    """" The hyper params to configure the training. See Supplementary C. "implementation details".
         Note: For backward compatibility, some of the variable names here are different than those mentioned in the
         paper. We explain in comments, how each variable is referenced at the paper.
    """

    metrics: MetricsCfg


    batch_size: int = 2048
    """ batch size """

    lr: float = 0.003
    """ initial learning rate """

    max_epoch: int = 5
    """ max number of epochs """

    alternate_ys: int = 21
    """ Whether and how to use alternate training. 0: no alternation|12: object then attr|21: attr then obj"""


    lr_step2: float = 3e-05  #
    """ Step 2 initial learning rate. Only relevant if alternate_ys != 0 """

    max_epoch_step2: int = 1000
    """ Step 2 max number of epochs. Only relevant if alternate_ys != 0 """

    weight_decay: float = 0.1
    """ weight_decay """

    HSIC_coeff: float = 10
    """ \lambda_rep in paper """

    alphaH: float = 0
    """ \lambda_oh in paper """

    alphaH_step2 = -1
    """ Step 2 \lambda_oh. Only relevant if alternate_ys != 0. If set to -1, then take --alphaH value """

    lambda_CE: float = 1
    """ a coefficient for L_data """

    lambda_feat: float = 1
    """ \lambda_ao in paper """

    lambda_ao_emb: float = 0
    """ \lambda_ao when projecting pretrained image features to the feature space \X 
    (as explained for Zappos in section C.1)
    Note: --lambda_feat and --lambda_ao_emb cant be both non-zero (we raise exception for this case).  
    """

    lambda_aux_disjoint: float = 100
    """ \lambda_icore in paper """

    lambda_aux_img: float = 10
    """ \lambda_ig in paper, when --lambda_feat>0"""

    lambda_aux: float = 0
    """ \lambda_ig in paper, when --lambda_ao_emb>0"""

    mu_img_feat: float = 0.1
    """ \lambda_ao at inference time """

    balanced_loss: bool = True
    """ Weighed the loss of ||φa−ha||^2 and ||φo−ho||^2 according to the respective attribute and object frequencies in 
        the training set (Described in supplementary C.2). """

    triplet_loss_margin: float = 0.5
    """ The margin for the triplet loss. Same value as used by attributes-as-operators """

    optimizer_name: str = 'Adam'

    seed: int = 0
    """ random seed """

    test_batchsize: int = -1
    """batch-size for inference; default uses the training batch size"""

    verbose: bool = True
    num_workers: int = 8
    shuffle_eval_set: bool = True
    n_iter: int = field(init=False)
    mu_disjoint: float = field(init=False)
    mu_ao_emb: float = field(init=False)
    primary_early_stop_metric: EarlyStopMetric = field(init=False)
    freeze_class1: bool = field(init=False)
    freeze_class2: bool = field(init=False)
    Y12_balance_coeff: float = field(init=False)


    def __post_init__(self):
        # sanity checks (assertions)
        assert (self.alternate_ys in [0, 12, 21])
        assert not ((self.lambda_ao_emb > 0) and (self.lambda_feat > 0))
        if self.lambda_feat == 0:
            assert(self.mu_img_feat == 0)

        # assignments
        if self.test_batchsize <= 0:
            self.test_batchsize = self.batch_size
        if self.alphaH_step2 < 0:
            self.alphaH_step2 = self.alphaH

        self.mu_disjoint = self.lambda_CE
        self.mu_ao_emb = self.lambda_ao_emb
        self.primary_early_stop_metric = EarlyStopMetric('epoch', 'max')
        self.Y12_balance_coeff = 0.5
        self.freeze_class1 = False
        self.freeze_class2 = False
        self.n_iter = -1  # Should be updated after data is loaded



    def set_n_iter(self, num_train_samples, max_epoch = None):
        if max_epoch is None:
            max_epoch = self.max_epoch
        self.n_iter = int((max_epoch) * np.ceil(num_train_samples / self.batch_size))

    def __getitem__(self, key):
        """ Allow accessing instance attributes as dictionary keys """
        return getattr(self, key)

@dataclass
class ExperimentCfg():
    output_dir: str = field(alias="-o")
    ignore_existing_output_contents: bool = True
    gpu: int = 0
    use_wandb: bool = True
    wandb_user: str = 'none'
    project_name: str = 'causal_comp_prep'
    experiment_name: str = 'default'
    instance_name: str = 'default'
    git_hash: str = ''
    sync_uid: str = ''
    report_imbalanced_metrics: bool = False

    # float precision when logging to CSV file
    csv_precision: int = 8

    delete_dumped_preds: bool = True


    def __post_init__(self):

        # Set default experiment name
        self._set_default_experiment_name()


    def _set_default_experiment_name(self):
        at_ngc: bool = ('NGC_JOB_ID' in os.environ.keys())
        at_docker = np.in1d(['/opt/conda/bin'], np.array(sys.path))[0]
        at_local_docker = at_docker and not at_ngc
        name_suffix = '_local'
        if at_local_docker:
            name_suffix += '_docker'
        elif at_ngc:
            name_suffix = '_ngc'
        if self.experiment_name == 'default':
            self.experiment_name = 'dev' + name_suffix
        if self.instance_name == 'default':
            self.instance_name = 'dev' + name_suffix

    def __getitem__(self, key):
        """ Allow accessing instance attributes as dictionary keys """
        return getattr(self, key)


@dataclass
class CommandlineArgs():
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    exp: ExperimentCfg
    device: str = 'cuda'

    @classmethod
    def get_args(cls):
        args: cls = to_simple_parsing_args(cls)
        return args
