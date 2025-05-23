# TabDDPM: Modelling Tabular Data with Diffusion Models
Differentially Private TabDDPM (DP-TabDDPM)
This repository is a modified implementation of TabDDPM, extended to support differential privacy using DP-SGD (Differentially Private Stochastic Gradient Descent). The goal is to enable privacy-preserving synthetic tabular data generation by integrating principled privacy guarantees directly into the training process.

Features:

✅ Full integration of DP-SGD for differentially private training

✅ Retains core functionality and architecture of the original TabDDPM

✅ Supports common DP accounting and hyperparameter tuning

✅ Useful for research in privacy-preserving machine learning and synthetic data generation


[//]: # ()
[//]: # (This is the official code for our paper "TabDDPM: Modelling Tabular Data with Diffusion Models" &#40;[paper]&#40;https://arxiv.org/abs/2209.15421&#41;&#41;)

[//]: # ()
[//]: # (<!-- ## Results)

[//]: # (You can view all the results and build your own tables with this [notebook]&#40;notebooks/Reports.ipynb&#41;. -->)

[//]: # ()
[//]: # (## Setup the environment)

[//]: # (1. Install [conda]&#40;https://docs.conda.io/en/latest/miniconda.html&#41; &#40;just to manage the env&#41;.)

[//]: # (2. Run the following commands)

[//]: # (    ```bash)

[//]: # (    export REPO_DIR=/path/to/the/code)

[//]: # (    cd $REPO_DIR)

[//]: # ()
[//]: # (    conda create -n tddpm python=3.9.7)

[//]: # (    conda activate tddpm)

[//]: # ()
[//]: # (    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html)

[//]: # (    pip install -r requirements.txt)

[//]: # ()
[//]: # (    # if the following commands do not succeed, update conda)

[//]: # (    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR})

[//]: # (    conda env config vars set PROJECT_DIR=${REPO_DIR})

[//]: # ()
[//]: # (    conda deactivate)

[//]: # (    conda activate tddpm)

[//]: # (    ```)

[//]: # ()
[//]: # (## Running the experiments)

[//]: # ()
[//]: # (Here we describe the neccesary info for reproducing the experimental results.  )

[//]: # (Use `agg_results.ipynb` to print results for all dataset and all methods.)

[//]: # ()
[//]: # (### Datasets)

[//]: # ()
[//]: # (We upload the datasets used in the paper with our train/val/test splits &#40;link below&#41;. We do not impose additional restrictions to the original dataset licenses, the sources of the data are listed in the paper appendix. )

[//]: # ()
[//]: # (You could load the datasets with the following commands:)

[//]: # ()
[//]: # (``` bash)

[//]: # (conda activate tddpm)

[//]: # (cd $PROJECT_DIR)

[//]: # (wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar)

[//]: # (tar -xvf data.tar)

[//]: # (```)

[//]: # ()
[//]: # (### File structure)

[//]: # (`tab-ddpm/` -- implementation of the proposed method  )

[//]: # (`tuned_models/` -- tuned hyperparameters of evaluation model &#40;CatBoost or MLP&#41;)

[//]: # ()
[//]: # (All main scripts are in `scripts/` folder:)

[//]: # ()
[//]: # (- `scripts/pipeline.py` are used to train, sample and eval TabDDPM using a given config  )

[//]: # (- `scripts/tune_ddpm.py` -- tune hyperparameters of TabDDPM)

[//]: # (- `scripts/eval_[catboost|mlp|simple].py` -- evaluate synthetic data using a tuned evaluation model or simple models)

[//]: # (- `scripts/eval_seeds.py` -- eval using multiple sampling and multuple eval seeds)

[//]: # (- `scripts/eval_seeds_simple.py` --  eval using multiple sampling and multuple eval seeds &#40;for simple models&#41;)

[//]: # (- `scripts/tune_evaluation_model.py` -- tune hyperparameters of eval model &#40;CatBoost or MLP&#41;)

[//]: # (- `scripts/resample_privacy.py` -- privacy calculation  )

[//]: # ()
[//]: # (Experiments folder &#40;`exp/`&#41;:)

[//]: # (- All results and synthetic data are stored in `exp/[ds_name]/[exp_name]/` folder)

[//]: # (- `exp/[ds_name]/config.toml` is a base config for tuning TabDDPM)

[//]: # (- `exp/[ds_name]/eval_[catboost|mlp].json` stores results of evaluation &#40;`scripts/eval_seeds.py`&#41;  )

[//]: # ()
[//]: # (To understand the structure of `config.toml` file, read `CONFIG_DESCRIPTION.md`.)

[//]: # ()
[//]: # (Baselines:)

[//]: # (- `smote/`)

[//]: # (- `CTGAN/` -- TVAE [official repo]&#40;https://github.com/sdv-dev/CTGAN&#41;)

[//]: # (- `CTAB-GAN/` --  [official repo]&#40;https://github.com/Team-TUD/CTAB-GAN&#41;)

[//]: # (- `CTAB-GAN-Plus/` -- [official repo]&#40;https://github.com/Team-TUD/CTAB-GAN-Plus&#41;)

[//]: # ()
[//]: # (### Examples)

[//]: # ()
[//]: # (<ins>Run TabDDPM tuning.</ins>   )

[//]: # ()
[//]: # (Template and example &#40;`--eval_seeds` is optional&#41;: )

[//]: # (```bash)

[//]: # (python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds)

[//]: # (python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds)

[//]: # (```)

[//]: # ()
[//]: # (<ins>Run TabDDPM pipeline.</ins>   )

[//]: # ()
[//]: # (Template and example  &#40;`--train`, `--sample`, `--eval` are optional&#41;: )

[//]: # (```bash)

[//]: # (python scripts/pipeline.py --config [path_to_your_config] --train --sample --eval)

[//]: # (python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample)

[//]: # (```)

[//]: # (It takes approximately 7min to run the script above &#40;NVIDIA GeForce RTX 2080 Ti&#41;.  )

[//]: # ()
[//]: # (<ins>Run evaluation over seeds</ins>   )

[//]: # (Before running evaluation, you have to train the model with the given hyperparameters &#40;the example above&#41;.  )

[//]: # ()
[//]: # (Template and example: )

[//]: # (```bash)

[//]: # (python scripts/eval_seeds.py --config [path_to_your_config] [n_eval_seeds] [ddpm|smote|ctabgan|ctabgan-plus|tvae] synthetic [catboost|mlp] [n_sample_seeds])

[//]: # (python scripts/eval_seeds.py --config exp/churn2/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5)

[//]: # (```)