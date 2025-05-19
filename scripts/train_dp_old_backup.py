import sys

# sys.path.append('..')
# sys.path.append('C:\\Users\\fjun\\Desktop\\tab-ddpm-main\\src\\opacus')
from copy import deepcopy
import torch
import os
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

from opacus import PrivacyEngine


class Trainer:
    def __init__(self, diffusion, train_iter, optimizer, lr, weight_decay, steps, DELTA, EPSILON, privacy_engine,
                 device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = optimizer
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 100
        self.ema_every = 1000
        self.DELTA = DELTA
        self.EPSILON = EPSILON
        self.privacy_engine = privacy_engine
        # self.diff2_loss = diff2

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion._module.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:

            # x, out_dict = next(self.train_iter)

            i, data = next(enumerate(self.train_iter))
            x, out_dict = data

            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                ######
                epsilon = self.privacy_engine.get_epsilon(self.DELTA)
                print(epsilon, self.EPSILON)
                # if epsilon > self.EPSILON:
                #     break
                # print(
                #     f"(ε = {epsilon:.2f}, δ = {self.DELTA})"
                # )
                ######
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                    print(f"(ε = {epsilon:.2f}, δ = {self.DELTA})")
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


def train_dp_old(
        parent_dir,
        real_data_path='data/higgs-small',
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        device=torch.device('cuda:1'),
        seed=0,
        change_val=False,
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    # diff2 = GaussianMultinomialDiffusion(
    #     num_classes=K,
    #     num_numerical_features=num_numerical_features,
    #     denoise_fn=model,
    #     gaussian_loss_type=gaussian_loss_type,
    #     num_timesteps=num_timesteps,
    #     scheduler=scheduler,
    #     device=device
    # )
    diffusion.train()

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    # train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_dp_dataloader(dataset, split='train', batch_size=batch_size)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)

    privacy_engine = PrivacyEngine()

    dataset_len = len(dataset.X_num['train'])
    epochs = steps / (dataset_len / batch_size)


    DELTA = 1e-05
    NOISE_MULTIPLIER = 0.1 # not used in make_private_with_epsilon
    EPSILON = 10
    # MAX_GRAD_NORM = 10  # links unten
    # MAX_GRAD_NORM = 1 # links oben
    MAX_GRAD_NORM = 0.1  # rechts oben

    print("batch_size", batch_size)
    print("dataset len: ", dataset_len)
    print("Steps: ", steps)
    print("Epochs: ", epochs)
    print("Target eps: ", EPSILON)
    print("Target delta: ", DELTA)
    print("Max grad norm: ", MAX_GRAD_NORM)

    diffusion, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=diffusion,
        optimizer=optimizer,
        data_loader=train_loader,
        target_delta=DELTA,
        target_epsilon=EPSILON,
        epochs=epochs,
        max_grad_norm=MAX_GRAD_NORM,
    )

    # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     target_delta=DELTA,
    #     target_epsilon=EPSILON,
    #     epochs=epochs,
    #     max_grad_norm=MAX_GRAD_NORM,
    # )

    # diffusion, optimizer, train_loader = privacy_engine.make_private(
    # module=diffusion,
    # optimizer=optimizer,
    # data_loader=train_loader,
    # max_grad_norm=MAX_GRAD_NORM,
    # noise_multiplier=NOISE_MULTIPLIER
    # )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    model.to(device)

    trainer = Trainer(
        diffusion,
        train_loader,
        optimizer,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        DELTA=DELTA,
        EPSILON=EPSILON,
        privacy_engine=privacy_engine
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))