#Code from https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

#dataset_path='/raid/ryan/CleanCode/Datasets/diff_rendering/non_triton/tugboat/renderings/spin_3600'
dataset_path='/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair/combined_photos'

device=torch.device('cuda:0')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).to(device)

trainer = Trainer(
    diffusion,
    dataset_path,
    train_batch_size = 4, #Originally was 32
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    results_folder = './results',
    save_and_sample_every=3000, #Default=1000
)

###############

#trainer.load(11)

###############

trainer.train()
