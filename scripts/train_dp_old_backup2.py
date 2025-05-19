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
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        # self.diffusion = diffusion
        self.ema_model = deepcopy(diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        # self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        # self.optimizer = optimizer
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000


        # For noise multiplicity:

        self.num_noise_levels = 8

        """
        MAKE PRIVATE
        START
        """
        self.DELTA = 1e-05
        self.EPSILON = 10

        NOISE_MULTIPLIER = 0.1  # automatically calculated by make_private_with_epsilon
        MAX_GRAD_NORM = 1  #

        epochs = steps / len(train_iter)
        print("Steps: ", steps)
        print("batches:", len(train_iter))
        print("Epochs: ", epochs)


        print("Target eps: ", self.EPSILON)
        print("Target delta: ", self.DELTA)
        print("Max grad norm: ", MAX_GRAD_NORM)

        self.privacy_engine = PrivacyEngine()
        self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private_with_epsilon(
            module=diffusion,
            optimizer=torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay),
            data_loader=train_iter,
            target_delta=self.DELTA,
            target_epsilon=self.EPSILON,
            epochs=epochs,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"Using sigma={self.optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
        """
        MAKE PRIVATE
        END
        """

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict, num_noise_levels):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()


        # loss_multi, loss_gauss = self.diffusion._module.mixed_loss(x, out_dict)
        """NOISE MULTIPLICITY START"""
        total_loss_multi = 0.0
        total_loss_gauss = 0.0
        for _ in range(num_noise_levels):
            loss_multi, loss_gauss = self.diffusion._module.mixed_loss(x, out_dict)
            total_loss_multi += loss_multi
            total_loss_gauss += loss_gauss

        # Average loss across multiple noise levels
        loss_multi = total_loss_multi / num_noise_levels
        loss_gauss = total_loss_gauss / num_noise_levels
        """NOISE MULTIPLICITY END"""


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

            # batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
            """NOISE MULTIPLICITY START"""
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict, num_noise_levels=self.num_noise_levels)
            """NOISE MULTIPLICITY END"""

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                # ######
                # epsilon = self.privacy_engine.get_epsilon(self.DELTA)
                # print(epsilon, self.EPSILON)

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

            if (step + 1) % self.print_every == 0:
                epsilon = self.privacy_engine.get_epsilon(self.DELTA)
                print(epsilon, self.EPSILON)


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
    model.to(device)

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
    diffusion.train()

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    # train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_dp_dataloader(dataset, split='train', batch_size=batch_size)

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))