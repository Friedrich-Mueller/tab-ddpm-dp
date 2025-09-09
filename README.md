# Differentially Private TabDDPM (DP-TabDDPM): Modelling Tabular Data with Differentially Private Diffusion Models
This repository is a modified implementation of [TabDDPM](https://github.com/yandex-research/tab-ddpm), extended to support differential privacy using DP-SGD (Differentially Private Stochastic Gradient Descent). All credit for the codebase goes to [TabDDPM](https://github.com/yandex-research/tab-ddpm). 

Features:

✅ Full integration of DP-SGD for differentially private training

✅ Retains core functionality and architecture of the original TabDDPM

✅ Supports common DP accounting and hyperparameter tuning

✅ Useful for research in privacy-preserving machine learning and synthetic tabular data generation


[//]: # ()
[//]: # (This is the official code for our paper "TabDDPM: Modelling Tabular Data with Diffusion Models" &#40;[paper]&#40;https://arxiv.org/abs/2209.15421&#41;&#41;)

[//]: # ()
[//]: # (<!-- ## Results)

[//]: # (You can view all the results and build your own tables with this [notebook]&#40;notebooks/Reports.ipynb&#41;. -->)

For , and [running the experiments](https://github.com/yandex-research/tab-ddpm?tab=readme-ov-file#running-the-experiments) outside of Differential Privacy, please refer to the original repo of TabDDPM.

For examples pointing out the additional differentially private options and how they are executed:


## Setup the environment
The setup is done completely analogue to the original TabDDPM. All library version are the same.
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

### Datasets

Nothing changed here. You can download their datasets and run them just as you could with TabDDPM. This is advisable at the very least to see how the data has to be prepared to be used with TabDDPM.

You could load the datasets with the following commands:

``` bash
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

### File structure
`tab-ddpm/` -- implementation of the proposed method  
`tuned_models/` -- tuned hyperparameters of evaluation model (CatBoost or MLP)

All main scripts are in `scripts/` folder:

- `scripts/pipeline.py` are used to train, sample and eval TabDDPM using a given config  
- `scripts/tune_ddpm.py` -- tune hyperparameters of TabDDPM
- `scripts/eval_[catboost|mlp|simple].py` -- evaluate synthetic data using a tuned evaluation model or simple models
- `scripts/eval_seeds.py` -- eval using multiple sampling and multuple eval seeds
- `scripts/eval_seeds_simple.py` --  eval using multiple sampling and multuple eval seeds (for simple models)
- `scripts/tune_evaluation_model.py` -- tune hyperparameters of eval model (CatBoost or MLP)
- `scripts/resample_privacy.py` -- privacy calculation  

Experiments folder (`exp/`):
- All results and synthetic data are stored in `exp/[ds_name]/[exp_name]/` folder
- `exp/[ds_name]/config.toml` is a base config for tuning TabDDPM
- `exp/[ds_name]/eval_[catboost|mlp].json` stores results of evaluation (`scripts/eval_seeds.py`)  

To understand the structure of `config.toml` file, read `CONFIG_DESCRIPTION.md`.

### Examples

#### DP-TabDDPM


<ins>Run DP-TabDDPM tuning.</ins>   

Either `--dp_eps [Int|Float]` or `--dp_noise [Int|Float]` is required.

Template and examples (`--eval_seeds` is optional):
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds --dp_eps [epsilon]
python scripts/tune_ddpm.py wilt 3096 synthetic catboost wilt_dp_eps_9 --eval_seeds --dp_eps 9
python scripts/tune_ddpm.py wilt 3096 synthetic catboost wilt_dp_eps_9 --eval_seeds --train_dp_noise 1.063232421875
```
Note how the first example will tune a model while maintaining differential privacy with given target epsilon (here $\epsilon$ = 9).<br>
Note how the second example will tune a model while maintaining a given target noise injection (here $\sigma$ = 1.063232421875).

<ins>Run DP-TabDDPM pipeline.</ins>   

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

(With the current config.toml in ../wilt_dp_eps_9_best a target $\epsilon$ = 9 results in of $\sigma$ = 1.063232421875 and vice versa.)  

#### TabDDPM

<ins>Run TabDDPM tuning.</ins>   

Template and examples (`--eval_seeds` is optional): 
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```


<ins>Run TabDDPM pipeline.</ins>   

Template and examples (`--train`, `--sample`, `--eval` are optional): 
```bash
python scripts/pipeline.py --config [path_to_your_config] --train --sample --eval
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample
```

<ins>Run evaluation over seeds</ins>   
Before running evaluation, you have to train the model with the given hyperparameters (the example above).  

Template and example: 
```bash
python scripts/eval_seeds.py --config [path_to_your_config] [n_eval_seeds] [ddpm|smote|ctabgan|ctabgan-plus|tvae] synthetic [catboost|mlp] [n_sample_seeds]
python scripts/eval_seeds.py --config exp/churn2/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
```


--- 

If you find any errors or run into problems, please don't hesitate reaching out to me.