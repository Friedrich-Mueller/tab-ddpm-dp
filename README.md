# Differentially Private TabDDPM (DP-TabDDPM): Modelling Tabular Data with Differentially Private Diffusion Models
This repository is a modified implementation of [TabDDPM](https://github.com/yandex-research/tab-ddpm), extended to support differential privacy using DP-SGD (Differentially Private Stochastic Gradient Descent). All credit for the codebase goes to [TabDDPM](https://github.com/yandex-research/tab-ddpm). 

Features:

✅ Full integration of DP-SGD for differentially private training

✅ Retains core functionality and architecture of the original TabDDPM

✅ Utilizes Noise Multiplicity as proposed by [Tim Dockhorn et al](https://arxiv.org/abs/2210.09929) 

✅ Supports common DP accounting and hyperparameter tuning

✅ Useful for research in privacy-preserving machine learning and synthetic tabular data generation


[//]: # ()
[//]: # (This is the official code for our paper "TabDDPM: Modelling Tabular Data with Diffusion Models" &#40;[paper]&#40;https://arxiv.org/abs/2209.15421&#41;&#41;)

[//]: # ()
[//]: # (<!-- ## Results)

[//]: # (You can view all the results and build your own tables with this [notebook]&#40;notebooks/Reports.ipynb&#41;. -->)

For examples pointing out the additional differentially private options and how they are executed, jump to [Examples - DP-TabDDPM](https://github.com/Friedrich-Mueller/tab-ddpm-dp?tab=readme-ov-file#examples---dp-tabddpm)

## Table of Contents

- [Setup the environment](#setup-the-environment)
- [Running the experiments](#running-the-experiments)
  - [Datasets](#datasets)
  - [File structure](#file-structure)
  - [Examples - DP-TabDDPM](#examples---dp-tabddpm)
    - [Run DP-TabDDPM tuning](#run-dp-tabddpm-tuning)
    - [Run DP-TabDDPM pipeline](#run-dp-tabddpm-pipeline)
    - [Example Results (wilt dataset)](#example-results-wilt-dataset)
    - [Example Results (churn2 embedded categoricals)](#example-results-churn2-embedded-categoricals)
  - [Examples - TabDDPM](#examples---tabddpm)
    - [Run TabDDPM tuning](#run-tabddpm-tuning)
    - [Run TabDDPM pipeline](#run-tabddpm-pipeline)
    - [Run evaluation over seeds](#run-evaluation-over-seeds)
- [Changes made compared to the TabDDPM repository](#changes-made-compared-to-the-tabddpm-repository)
- [Contact / Troubleshooting](#contact--troubleshooting)



## Setup the environment
The setup is done completely analogue to the original TabDDPM. All library version are the same. (Including an old version of Opacus)
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n tddpm-dp python=3.9.7
    conda activate tddpm-dp

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate tddpm-dp
    ```



## Running the experiments

Please note that if you want to run the complete experiments from the [TabDDPM paper](https://arxiv.org/abs/2209.15421), such as different models (CTAB-GAN, CTAB-GAN-Plus, etc) which were used for benchmarking, you will want to use the original TabDDPM repository.

DP-TabDDPM offers the complete TabDDPM Diffusion Model functionality and additionally the option to run it with the application of differential privacy via DP-SGD.

Beware that training under DP increases runtimes significantly. Additionally, the runtime grows linearly with the amount/value of Noise Multiplicity.

### Datasets

Nothing changed here. You can download their datasets and run them just as you could with TabDDPM. This is advisable at the very least to see how the data has to be prepared to be used with TabDDPM.

You can load the datasets with the following commands:

``` bash
conda activate tddpm-dp
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

### File structure

For the file structure, please refer to the [original repo](https://github.com/yandex-research/tab-ddpm?tab=readme-ov-file#file-structure)

### Examples - DP-TabDDPM

<a id="run-dp-tabddpm-tuning"></a>
#### <ins>Run DP-TabDDPM tuning</ins>   

Either `--dp_eps [Int|Float]` or `--dp_noise [Int|Float]` is required.

Template and examples (`--eval_seeds` is optional):
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds --dp_eps [epsilon]
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds --train_dp_noise [noise]
python scripts/tune_ddpm.py wilt 3096 synthetic catboost wilt_dp_eps_9 --eval_seeds --dp_eps 9
python scripts/tune_ddpm.py wilt 3096 synthetic catboost wilt_dp_eps_9 --eval_seeds --train_dp_noise 1.063232421875
```
Note how the first example will tune a model while maintaining differential privacy with given target epsilon (here $\epsilon$ = 9).<br>
Note how the second example will tune a model while maintaining a given target noise injection (here $\sigma$ = 1.063232421875).

<a id="run-dp-tabddpm-pipeline"></a>
#### <ins>Run DP-TabDDPM pipeline</ins>   

Either `--train_dp_eps [Int|Float]` or `--train_dp_noise [Int|Float]` is required.

Templates and examples (`--train`, `--sample`, `--eval` are optional, `--train_dp_eps x` is optional): 
```bash
python scripts/pipeline.py --config [path_to_your_config] --train_dp_eps  [Int|Float] --sample --eval
python scripts/pipeline.py --config [path_to_your_config] --train_dp_noise  [Int|Float] --sample --eval
python scripts/pipeline.py --config exp/wilt/wilt_dp_eps_9_best/config.toml --train_dp_eps 9 --sample --eval
python scripts/pipeline.py --config exp/wilt/wilt_dp_eps_9_best/config.toml --train_dp_noise 1.063232421875 --sample --eval
```
Note how the first example will train a model while maintaining differential privacy with given target epsilon (here $\epsilon$ = 9).<br>
Note how the second example will train a model while maintaining a given target noise injection (here $\sigma$ = 1.063232421875).

(With the current config.toml in ../wilt_dp_eps_9_best a target $\epsilon$ = 9 results in $\sigma$ = 1.063232421875 and vice versa.)  

#### Example Results (wilt dataset):

Here we see the results of the utility analysis for the Wilt dataset, which consists exclusively of continuous features, measured using the utility measure, a generic Catboost classifier.

The utility remains strong up to an epsilon of approximately 1, with an F1 score of 0.87.

Below an epsilon of 1, the utility drops sharply, accompanied by increasing noise.

The utility of the synthetic Wilt data is quite good compared to the original data, which reaches a baseline F1 score of approximately 0.9.
<p align="center">
  <img width="439" height="374" src="./imgs/wilt10-0.6.png">
</p>

#### Example Results (churn2 embedded categoricals):

Shown here are the results of the Churn Modeling dataset, consisting of 7 continuous and 4 categorical features.

The first column of the table shows that the vanilla model performs equally well on all three representations of the data — that is, on the original mixed-type data as well as on the two types of embeddings.

The first row illustrates how the fidelity and utility of the mixed-type representation already break down at epsilons of 1000 and 100, respectively.

As seen in the second row, the categorical embeddings significantly improve robustness.

This suggests that under DP it is more optimal to accept some information loss due to naive embeddings than to attempt to work with heterogeneous features.

And as you can see in the bottom row, a full Latent Space Representation achieves even better results with an F1 score of 0.57 at an epsilon of 5. That's only a 22% loss of utility.

<p align="center">
  <img width="683" height="216" src="./imgs/churn2_embeddings_results.png">
</p>



### Examples - TabDDPM

#### <ins>Run TabDDPM tuning</ins>   

Template and examples (`--eval_seeds` is optional): 
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```


#### <ins>Run TabDDPM pipeline</ins>   

Template and examples (`--train`, `--sample`, `--eval` are optional): 
```bash
python scripts/pipeline.py --config [path_to_your_config] --train --sample --eval
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample
```

#### <ins>Run evaluation over seeds</ins>   
Before running evaluation, you have to train the model with the given hyperparameters (the example above).  

Template and example: 
```bash
python scripts/eval_seeds.py --config [path_to_your_config] [n_eval_seeds] [ddpm|smote|ctabgan|ctabgan-plus|tvae] synthetic [catboost|mlp] [n_sample_seeds]
python scripts/eval_seeds.py --config exp/churn2/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
```

## Changes made compared to the [TabDDPM](https://github.com/rotot0/tab-ddpm) repository

- Added `train_dp_eps.py` and `train_dp_noise.py`, implementing the respective Opacus methods for training with Differential Privacy (DP-SGD).
  - Within these scripts:
    - Replaced `mixed_loss` with `mixed_loss_dp`, enabling per-sample loss calculation and noise multiplicity.
    - Replaced the custom dataloader `prepare_fast_dataloader` with `prepare_fast_dp_dataloader` to ensure compatibility with Opacus' `PrivacyEngine`.
    - Added several learning rate annealing options.
- Extended `pipeline.py` to support running `train_dp_eps.py` and `train_dp_noise.py`.
- Extended `tune_ddpm.py` to support running `train_dp_eps.py` and `train_dp_noise.py`.
- When running DP experiments, configuration files now require a `[train.dp]` subsection containing parameters for:
  - Learning rate annealing
  - Gradient clipping
  - Noise multiplicity
--- 

## Contact / Troubleshooting
If you find any errors or run into problems, please don't hesitate reaching out to me.