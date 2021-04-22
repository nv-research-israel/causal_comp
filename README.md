<p align="center"><img src="Causal.png" width="400"/></p>

# A causal view of compositional zero-shot recognition
This repository hosts the dataset and source code for the paper "A causal view of compositional zero-shot recognition". Yuval Atzmon, Felix Kreuk, Uri Shalit, Gal Chechik, NeurIPS 2020 (Spotlight)


## Code 

### Setup
#### Build docker image:
```
cd docker_cfg
docker build --network=host -t causal_comp -f Dockerfile .
cd ..
```


#### Prepare UT Zappos50KUT with TMN split:
To reproduce Zappos50KUT results according to TMN evaluation split, you should prepare the dataset according to 
taskmodularnets:
https://github.com/facebookresearch/taskmodularnets

The Zappos50KUT dataset is for academic, non-commercial use only, released by:

* A. Yu and K. Grauman. "Fine-Grained Visual Comparisons with Local Learning". In CVPR, 2014.
* A. Yu and K. Grauman. "Semantic Jitter: Dense Supervision for Visual Comparisons via Synthetic Images". In ICCV, 2017.


#### Reproduce Zappos experiment. 
Notes: 
1. Set `DATA_DIR` to the directory containing the data.
2. Set `SEED` to [0..4]
3. Set `SOURCE_CODE_DIR` to the current project workdir
4. Set `OUTPUT_DIR` to the directory to save the result

```
SEED=0  # set seed \in [0..4]
SOURCE_CODE_DIR=$HOME/git/causal_comp_prep/
DATA_DIR=/local_data/zap_tmn # SET HERE THE DATA DIR
DATASET=zap_tmn
OUTPUT_DIR=/tmp/output/causal_comp_${DATASET}__seed${SEED}

# prepare output directory
mkdir -p ${OUTPUT_DIR} 
rm -r ${OUTPUT_DIR}/* 
# run experiment
docker run --net=host -v ${SOURCE_CODE_DIR}:/workspace/causal_comp -v ${DATA_DIR}:/data/zap_tmn -v ${OUTPUT_DIR}:/output --user $(id -u):$(id -g)  --shm-size=1g --ulimit memlock=-1 --ulimit stack=6710886  --rm -it causal_comp /bin/bash -c "cd /workspace/causal_comp/; python experiment_zappos_TMNsplit.py --seed=${SEED} --output_dir=/output --data_dir=/data/zap_tmn --use_wandb=0"
```

#### Reproduce AO-CLEVr experiments. 
Hyperparams for reproducing the results will be published soon


## AO-CLEVr Dataset

AO-CLEVr is a new synthetic-images dataset containing images of "easy" Attribute-Object categories, based on the CLEVr framework (Johnson et al. CVPR 2017). AO-CLEVr has attribute-object pairs created from 8 attributes: \{ red, purple, yellow, blue, green, cyan, gray, brown \} and 3 object shapes \{sphere, cube, cylinder\}, yielding 24 attribute-object pairs. Each pair consists of 7500 images. Each image has a single object that consists of the attribute-object pair. The object is randomly assigned one of two sizes (small/large), one of two materials (rubber/metallic), a random position, and random lightning according to CLEVr defaults. 

![Examples of AO-CLEVr images](./AO_CLEVr_examples.png)

The dataset can be accessed from [the following url](https://drive.google.com/drive/folders/1BBwW9VqzROgJXmvnfXcOxbLob8FB_jLf).


## Cite the paper
If you use the contents of this project, please cite our paper.

    @inproceedings{neurips2020_causal_comp_atzmon,
     author = {Atzmon, Yuval and Kreuk, Felix and Shalit, Uri and Chechik, Gal},
     booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
     title = {A causal view of compositional zero-shot recognition},
     year = {2020}
    }

For business inquiries, please contact [researchinquiries@nvidia.com](researchinquiries@nvidia.com) <br>
For press and other inquiries, please contact Hector Marinez at [hmarinez@nvidia.com](hmarinez@nvidia.com)
