# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

from pathlib import Path

from dataclasses import dataclass
from simple_parsing import ArgumentParser, ConflictResolution

from params import CommandlineArgs
import shlex

from main import main
from useful_utils import to_simple_parsing_args


@dataclass
class ScriptArgs():
    output_dir: str = '/tmp/output_results_dir'
    """ Directory to save the results """

    data_dir: str = '/local_data/zap_tmn'
    """ Directory that holds the data"""

    seed: int = 0
    """ Random seed. For zappos, choose \in [0..4] """

    use_wandb: bool = True
    """ log results to w&b """

    wandb_user: str = 'none'

    @classmethod
    def get_args(cls):
        args: cls = to_simple_parsing_args(cls)
        return args

if __name__ == '__main__':
    script_args = ScriptArgs.get_args()

    # Seed=0: Unseen=27.4, Seen=33.9, Harmonic=30.3, Closed=56.7, AUC=25.2

    # Set commandline arguments for the experiment.
    #  see params.py for their documentation and meaning with respect to definitions in the paper

    experiment_commandline_opt_string = f"""--output_dir={script_args.output_dir} --data_dir={script_args.data_dir} 
    --use_wandb={script_args.use_wandb}  --wandb_user={script_args.wandb_user}
    --learn_embeddings=1 --nlayers_label=0 --nlayers_joint_ao=2 --h_dim=300 --E_num_hidden=2 --E_num_common_hidden=1 
    --dataset_name=zap_tmn --dataset_variant=irrelevant --optimizer_name=adam --lr=0.0003 --max_epoch=1000 --alternate_ys=0 
    --weight_decay=5e-05 --calc_AUC=1 --lambda_feat=0 --mu_img_feat=0 --lambda_aux_disjoint=100 --lambda_aux_img=0 --metadata_from_pkl=0 
    --lambda_ao_emb=1000 --lambda_aux=0 --seed={script_args.seed} --alphaH=450.0 --HSIC_coeff=0.0045 --balanced_loss=0 --num_split=0
    --num_workers=8 --report_imbalanced_metrics=True"""
    if script_args.use_wandb:
        experiment_commandline_opt_string += """ --instance_name=dev_zap_tmn --project_name=causal_comp_prep_zap_tmn"""

    parser = ArgumentParser(conflict_resolution=ConflictResolution.NONE)
    parser.add_arguments(CommandlineArgs, dest='cfg')
    main_args: CommandlineArgs = parser.parse_args(shlex.split(experiment_commandline_opt_string)).cfg
    main(main_args)

