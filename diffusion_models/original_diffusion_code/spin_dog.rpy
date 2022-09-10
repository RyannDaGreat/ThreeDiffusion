# Code from https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import icecream

# dataset_path='/raid/ryan/CleanCode/Datasets/diff_rendering/non_triton/tugboat/renderings/spin_3600'
dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair/combined_photos"

device = torch.device("cuda:0")
torch.cuda.set_device(device)


resolution=128

def modify_prediction(image):
    #image = rotate_image(image, 3)
    image = crop_image(image, resolution, resolution, origin="center")
    return image


def modify_predictions(images):
    images = [modify_prediction(image) for image in images]
    display_image_on_macmax(tiled_images(images))
    return images

if 1 or "model" not in dir():
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)

if 1 or "diffusion" not in dir():
    diffusion = GaussianDiffusion(
        model,
        image_size=resolution,
        timesteps=1000,  # number of steps
        #sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=250,
        objective="pred_x0",  # We wanna use this, not noise...make my life easier lol...dont have to worry about messing the math up. Modify the model_predictions function
        loss_type="l1",  # L1 or L2
        modify_predictions=modify_predictions,
    ).to(device)

if 1 or "trainer" not in dir():
    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size=4,  # Originally was 32
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        augment_horizontal_flip=False,
        results_folder="./results",
        save_and_sample_every=2500,  # Default=1000
    )


def get_latest_milestone() -> int:
    if is_empty_folder("results"):
        return None
    return int(
        [
            get_file_name(x, include_file_extension=False).split("-")[1]
            for x in get_all_files(
                "results",
                just_file_names=True,
                file_extension_filter="pt",
                sort_by="date",
            )
        ][-1]
    )


def tiled_torch_images(images):
    images = rp.as_numpy_images(images)
    images = tiled_images(images)
    return images


###############

if get_latest_milestone() is not None:
    print(end='Loading checkpoint...')
    trainer.load(get_latest_milestone())
    print('done.')

###############

icecream.ic(device, dataset_path, rp.get_current_directory())

# trainer.ema.ema_model.train()
# trainer.train()

with torch.no_grad():
    trainer.ema.ema_model.eval()
    i = trainer.ema.ema_model.sample()