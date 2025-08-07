from copy import deepcopy
import torch
import os
import numpy as np
import zero
import math

from scripts.train import train
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

from opacus import PrivacyEngine

def print_grad_stats(model, name="Before Opacus"):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))  # Flatten gradients
    if grads:
        grads = torch.cat(grads)
        print(f"[{name}] Gradients - Mean: {grads.mean().item():.6f}, Std: {grads.std().item():.6f}, Max: {grads.max().item():.6f}, Min: {grads.min().item():.6f}")


class Trainer:
    def __init__(self, diffusion, train_iter, batch_size,  max_grad_norm, noise_multiplicity, dataset_len, lr, lr_anneal, weight_decay, steps, device=torch.device('cuda:1')):
        # self.diffusion = diffusion
        self.ema_model = deepcopy(diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        # self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        # self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])

        if lr_anneal == "none":
            # No annealing: do nothing on each step
            self._anneal_lr = lambda step: None
        elif lr_anneal == "linear":
            self._anneal_lr = self._anneal_lr_linear
        elif lr_anneal == "cosine":
            self._anneal_lr = self._anneal_lr_cosine
        elif lr_anneal == "flat_then_decay":
            self._anneal_lr = self._anneal_lr_flat_sharp_decay
        else:
            raise ValueError(f"Unknown lr_anneal method: {lr_anneal}")

        self.log_every = 10
        self.print_every = 50
        self.ema_every = 1000

        self.delta = dataset_len**-1

        self.noise_multiplicity_K = noise_multiplicity #50 both yield results

        """
        MAKE PRIVATE
        START
        """
        # self.noise_multiplier = 1.0239028930664062 # make_private_with_epsilon with eps 500



        # self.noise_multiplier = 0 # no privacy
        # self.max_grad_norm = 1e6 # no privacy
        # self.noise_multiplier = 0.25 # wilt
        # self.max_grad_norm = 0.2 # wilt
        # self.max_grad_norm = 0
        self.max_grad_norm = max_grad_norm
        # lr = 0.005

        # self.noise_multiplier = 0.1 # adult
        # self.max_grad_norm = 0.2 # adult


        # self.noise_multiplier = 0.1  # old dp settings
        # self.max_grad_norm = 0.01  # old dp settings

        # enter PrivacyEngine

        print("Dataset size(len(train_iter)*batch size), dataset_len: ", len(train_iter)*batch_size, dataset_len)
        self.privacy_engine = PrivacyEngine(accountant="rdp")
        self.target_epsilon = 9
        epochs = steps / (dataset_len / batch_size)
        self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private_with_epsilon(
            module=diffusion,
            optimizer=torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay),
            data_loader=train_iter,
            target_delta=self.delta,
            target_epsilon=self.target_epsilon,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm,
        )
        print(f"target eps={self.target_epsilon}\n"
              f"sigma={self.optimizer.noise_multiplier}\n"
              f"C={self.max_grad_norm}\n"
              f"lr: {self.init_lr}\n"
              f"lr_anneal: {lr_anneal}\n"
              f"noise_multiplicity_K: {self.noise_multiplicity_K}")
        """
        MAKE PRIVATE
        END
        """

        # self.diffusion = diffusion
        # self.optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        # self.train_iter = train_iter

    # linear decay
    def _anneal_lr_linear(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # cosine decay
    def _anneal_lr_cosine(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * 0.5 * (1 + math.cos(math.pi * frac_done))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # Flat for 80%, then linear decay
    def _anneal_lr_flat_sharp_decay(self, step):
        frac_done = step / self.steps
        if frac_done < 0.8:
            lr = self.init_lr
        else:
            lr = self.init_lr * (1 - (frac_done - 0.8) / 0.2)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr



    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()

        total_loss_multi = 0.0
        total_loss_gauss = 0.0
        for i in range(self.noise_multiplicity_K):
            # loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
            loss_multi, loss_gauss = self.diffusion._module.mixed_loss(x, out_dict)
            total_loss_multi += loss_multi
            total_loss_gauss += loss_gauss

        loss = (total_loss_multi / self.noise_multiplicity_K) + (total_loss_gauss / self.noise_multiplicity_K)
        loss.backward()

        # After loss.backward():

        # per_sample_grad_norms = []
        # for param in self.diffusion.parameters():
        #     if hasattr(param, 'grad_sample'):
        #         grad_sample = param.grad_sample.reshape(param.grad_sample.shape[0], -1)
        #         norms = torch.norm(grad_sample, dim=1)
        #         per_sample_grad_norms.append(norms)
        #
        # per_sample_grad_norms = torch.stack(per_sample_grad_norms, dim=1)
        # total_per_sample_norms = torch.norm(per_sample_grad_norms, dim=1)
        # clipping_ratio = (total_per_sample_norms > self.max_grad_norm).float().mean().item()
        #
        # print(f"[DEBUG] Clipping ratio: {clipping_ratio:.4f}")

        # Print raw gradients (before Opacus modifies them)
        # print_grad_stats(self.diffusion._module, "Before Opacus")

        self.optimizer.step()

        # Print gradients after Opacus has modified them
        # print_grad_stats(self.diffusion._module, "After Opacus")

        return (total_loss_multi / self.noise_multiplicity_K), (total_loss_gauss / self.noise_multiplicity_K)

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:

            # Get the next batch of data
            # x, out_dict = next(self.train_iter)
            i, data = next(enumerate(self.train_iter))
            x, out_dict = data

            out_dict = {'y': out_dict}

            # Run a training step
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            # Anneal learning rate as per your schedule
            self._anneal_lr(step)

            # Update current batch statistics
            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            # Logging after every log_every steps
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


            epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            # **Added: Print privacy statistics**
            if (step + 1) % self.print_every == 0:
                epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
                print(f"Step {step + 1}: Privacy ε = {epsilon:.2f}, δ = {self.delta}")

            # if epsilon > self.target_epsilon:
            #     print("TARGET EPSILON REACHED. STOPPING.")
            #     print(f"Step {step+1}: Privacy ε = {epsilon:.2f}, δ = {self.delta}")
            #     break


def train_dp_eps(
        parent_dir,
        real_data_path='data/higgs-small',
        steps=1000,
        lr=0.002,
        lr_anneal='linear',
        weight_decay=1e-4,
        batch_size=1024,
        max_grad_norm=0.4,
        noise_multiplicity=1,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        device=torch.device('cuda:1'),
        seed=0,
        change_val=False
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

    dataset_len = len(dataset.X_num['train'])
    epochs = steps / (dataset_len / batch_size)
    print("batch_size", batch_size)
    print("dataset len: ", dataset_len)
    print("Steps: ", steps)
    print("Epochs: ", epochs)

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
    # train_loader = lib.prepare_dp_dataloader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dp_dataloader(dataset, split='train', batch_size=batch_size)

    trainer = Trainer(
        diffusion,
        train_loader,
        dataset_len=dataset_len,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        noise_multiplicity=noise_multiplicity,
        lr=lr,
        lr_anneal=lr_anneal,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))