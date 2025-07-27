import subprocess
import lib
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

pipeline = f'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 9
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):

    ### DEFAULT
    # lr = trial.suggest_loguniform('lr', 0.00001, 0.01)
    # d_layers = _suggest_mlp_layers(trial)
    # weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    # steps = trial.suggest_categorical('steps', [100, 200, 300, 400, 500])
    # # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [50, 100, 500, 1000])
    # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # max_grad_norm = trial.suggest_loguniform("max_grad_norm", 0.05, 2.0)

    ### WILT eps 10
    # # lr = trial.suggest_loguniform('lr', 0.001, 0.01)
    # lr = trial.suggest_float('lr', 0.004, 0.006)
    # # lr_anneal = trial.suggest_categorical('lr_anneal', ['none', 'linear', 'cosine', 'flat_then_decay'])
    # lr_anneal = trial.suggest_categorical('lr_anneal', ['cosine'])
    # # d_layers = _suggest_mlp_layers(trial)
    # weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # steps = trial.suggest_categorical('steps', [400])
    # # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    # # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # # max_grad_norm = trial.suggest_loguniform("max_grad_norm", 0.1, 1.5)
    # max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1)
    # noise_multiplicity = trial.suggest_categorical("noise_multiplicity", [32])
    # # noise_multiplicity = trial.suggest_int("noise_multiplicity", 8, 35)


    # # ### WILT eps 8
    # lr = trial.suggest_loguniform('lr', 0.001, 0.01)
    # # lr_anneal = trial.suggest_categorical('lr_anneal', ['none', 'linear', 'cosine', 'flat_then_decay'])
    # lr_anneal = trial.suggest_categorical('lr_anneal', ['cosine'])
    # # d_layers = _suggest_mlp_layers(trial)
    # weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # steps = trial.suggest_categorical('steps', [400])
    # # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    # # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # max_grad_norm = trial.suggest_loguniform("max_grad_norm", 0.1, 1.5)
    # # noise_multiplicity = trial.suggest_categorical("noise_multiplicity", [8, 16, 32])
    # noise_multiplicity = trial.suggest_int("noise_multiplicity", 1, 50)

    # # ### WILT eps 10
    # # lr = trial.suggest_float('lr', 0.003, 0.006)
    # lr = trial.suggest_categorical('lr', [0.00436436456821403])
    # # lr_anneal = trial.suggest_categorical('lr_anneal', ['none', 'linear', 'cosine', 'flat_then_decay'])
    # lr_anneal = trial.suggest_categorical('lr_anneal', ['cosine'])
    # # d_layers = _suggest_mlp_layers(trial)
    # weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # steps = trial.suggest_categorical('steps', [400])
    # # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    # # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # # max_grad_norm = trial.suggest_float("max_grad_norm", 0.4, 0.7)
    # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.46165049332108])
    # # noise_multiplicity = trial.suggest_categorical("noise_multiplicity", [16])
    # noise_multiplicity = trial.suggest_int("noise_multiplicity", 10, 20)

    # ### WILT eps 50
    # lr = trial.suggest_float('lr', 0.003, 0.006)
    lr = trial.suggest_categorical('lr', [0.00436436456821403])
    # lr_anneal = trial.suggest_categorical('lr_anneal', ['none', 'linear', 'cosine', 'flat_then_decay'])
    lr_anneal = trial.suggest_categorical('lr_anneal', ['cosine'])
    # d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0
    batch_size = trial.suggest_categorical('batch_size', [256])
    steps = trial.suggest_categorical('steps', [400])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # max_grad_norm = trial.suggest_float("max_grad_norm", 0.4, 0.7)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.46165049332108])
    # noise_multiplicity = trial.suggest_categorical("noise_multiplicity", [16])
    noise_multiplicity = trial.suggest_int("noise_multiplicity", 95, 150)


    ### WILT NOISE MULTIPLICITY
    # lr = trial.suggest_categorical('lr', [0.00436436456821403])
    # # anneal = trial.suggest_categorical('anneal', ['linear', 'cosine', 'flat_then_decay'])
    # # lr_anneal = trial.suggest_categorical('lr_anneal', ['cosine', 'flat_then_decay'])
    # # d_layers = _suggest_mlp_layers(trial)
    # weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # steps = trial.suggest_categorical('steps', [400])
    # # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    # # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.46165049332108])
    # noise_multiplicity = trial.suggest_categorical("noise_multiplicity", [4, 8, 16, 32])






    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['lr_anneal'] = lr_anneal
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['train']['main']['max_grad_norm'] = max_grad_norm
    base_config['train']['main']['noise_multiplicity'] = noise_multiplicity
    # base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    # base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train_dp_eps', '--change_val'], check=True)

    n_datasets = 5
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=7, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train_dp_eps', '--sample'], check=True)

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run(['python3.9', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)