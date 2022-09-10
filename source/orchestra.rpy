#SUMMARY
#    - We use the blender datasets provided by Plenoxel.
#    - We train on their test data, and test on their train data:
#          We train the diffusion model on the test images, because there are more of them.
#          We train the plenoxel model with the positions in the test data, because there are 200 of them
#          There are 200 images in the train datasets, and 600 in the test
#    - We train our diffusion model with 128x128 images, and have a batch size of 200 (the number of training samples)
#    - This process does the diffusion, and coordinates with the plenoxel model code by running it in separate processes.
# Diffusion Code from https://github.com/lucidrains/denoising-diffusion-pytorch

#IMPORTS
import rp
import os
import rp.web_evaluator as web_evaluator
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import icecream
import inspect

#SETUP ASSERTIONS
assert rp.get_current_directory().endswith('/diffusion_for_nerf/source')

#PATH SETTINGS
project_root='..' #We're in the 'source' folder of the project
dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair"
diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_lego_direct_128')
plenoxel_opt_folder = '/raid/ryan/CleanCode/Github/svox2/opt' #TODO: Move this into the project

#DIFFUSION SETTINGS
diffusion_device = torch.device("cuda:1")
plenoxel_device = torch.device("cuda:0")
resolution=128 #Later on, this should be detected from the diffusion_model_folder

#DERIVED PATHS:
combined_photo_folder = rp.path_join(dataset_path,'combined_photos') #For training the diffusion model
training_json_file = rp.path_join(dataset_path,'transforms_train.json')

#SETTINGS VALIDATION
assert rp.folder_exists(combined_photo_folder)
assert rp.folder_exists(diffusion_model_folder)
assert rp.folder_exists(project_root)
assert rp.folder_exists(dataset_path)
assert rp.folder_exists(plenoxel_opt_folder)
assert rp.file_exists(training_json_file)

#SETUP
torch.cuda.set_device(diffusion_device)
dataset_json=rp.load_json(training_json_file)
image_filenames=[rp.get_file_name()]

#MAX BATCH SIZE FOR RESOLUTIONS:
#    128x128 : 225
#    256x256 : ?
BATCH_SIZE=len(dataset_json['frames'])
icecream.ic(BATCH_SIZE)


#GENERAL HELPER FUNCTIONS
@rp.memoized
def get_macmax_client():
    client=web_evaluator.Client('172.27.78.46')
    return client

def display_image_on_macmax(image):
    #Display an image on the Mac Max

    #Allow str too
    if isinstance(image,str):
        assert rp.file_exists(image),image
        assert rp.is_image_file(image),image
    if rp.is_image_file(image):
        image=rp.load_image(image)
        
    assert rp.is_image(image),'display_image_on_macmax: Input is not an image!'
    try:
        return get_macmax_client().evaluate(
            "x=decode_image_from_bytes(x);display_image(x);frames.append(x)",
            x=rp.encode_image_to_bytes(image),
        )
    except Exception as e:
        rp.fansi_print('WebEval Failed: '+str(e),'red')

def tiled_torch_images(images):
    images = rp.as_numpy_images(images)
    images = rp.tiled_images(images)
    return images

def do_in_dir(func,folder):
    #Decorator for functions - uses SetCurrentDirectoryTemporarily
    def wrapper(*args,**kwargs):
        with rp.SetCurrentDirectoryTemporarily(folder):
            return func(*args,**kwargs)
    wrapper.signature=inspect.signature(func) #Better rp autocompletion
    return wrapper


#DIFFUSION FUNCTIONS
times=70*BATCH_SIZE
def modify_prediction(image):
    global times
    if times:
        print(times)
        times-=1
        #image = rotate_image(image, randint(360))
        #image = rotate_image(image, 0)
        #image = rotate_image(image, 30)
    image = rp.crop_image(image, resolution, resolution, origin="center")
    return image

def modify_predictions(images):
    images = [modify_prediction(image) for image in images]

    try:
        display_image_on_macmax(rp.tiled_images(images))
    except Exception as e:
        rp.fansi_print('ERROR: '+str(e),'red')

    return images

def get_latest_milestone() :
    if rp.is_empty_folder("results"):
        return None
    return int(
        [
            rp.get_file_name(x, include_file_extension=False).split("-")[1]
            for x in rp.get_all_files(
                "results",
                just_file_names=True,
                file_extension_filter="pt",
                sort_by="date",
            )
        ][-1]
    )


#PLENOXEL FUNCTIONS
@do_in_dir(plenoxel_opt_folder)
def launch_trainer(experiment_name, gpu_id, dataset_path, config_path):
    assert_right_plenoxel_conditions()
    #Be sure to 'conda activate plenoxel' and CD into /raid/ryan/CleanCode/Github/svox2/opt before calling!
    command='./launch.sh %s %i %s -c %s'
    command%=(
        experiment_name,
        gpu_id,
        dataset_path,
        config_path
    )
    
    #Creates something like:
    #    ./launch.sh first_experiment_chair__fast__run0 0 /raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair -c configs/ryan_syn.json
    rp.fansi_print('RUNNING: '+command,'green','bold')
    os.system(command)

@do_in_dir(plenoxel_opt_folder)
def get_checkpoint_path(experiment_name):
    assert_right_plenoxel_conditions()
    output='./ckpt/%s/ckpt.npz'    
    output%=experiment_name
    assert rp.path_exists(output)
    return output
    
def assert_right_plenoxel_conditions():
    assert rp.get_current_directory().endswith("/svox2/opt")

@do_in_dir(plenoxel_opt_folder)
def render_imgs_circle(checkpoint_path, dataset_path):
    assert_right_plenoxel_conditions()

    command = "python render_imgs_circle.py --blackbg --radius 2 %s %s"
    command %= (
        checkpoint_path,
        dataset_path,
    )

    rp.fansi_print("COMMAND: " + command, "green", "bold")

    os.system(command)

    out_path = rp.get_parent_directory(checkpoint_path)
    out_path = rp.path_join(out_path, "circle_renders.mp4")
    assert rp.path_exists(out_path), "Something went wrong - the video wasnt written"
    
@do_in_dir(plenoxel_opt_folder)
def render_imgs(checkpoint_path, dataset_path):
    assert_right_plenoxel_conditions()

    command = "python render_imgs.py --blackbg --no_vid --no_lpips --train %s %s"
    command = "python render_imgs.py --blackbg --no_vid --no_lpips %s %s"
    command %= (
        checkpoint_path,
        dataset_path,
    )

    rp.fansi_print("COMMAND: " + command, "green", "bold")

    os.system(command)

    #out_path = get_parent_directory(checkpoint_path)
    #out_path = path_join(out_path, "circle_renders.mp4")
    #assert path_exists(out_path), "Something went wrong - the video wasnt written"
    
@do_in_dir(plenoxel_opt_folder)
def test_launch_trainer(train=True,images=True,video=True):
    rp.tic()
    
    experiment_name='first_experiment_chair__fast__subset__run1'
    experiment_name='lego__fast__128_black__run1'
    
    gpu_id=0
    
    #dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair'
    #dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair/subsets'
    dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/subsets/subset_25__just_train'
    dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/128x128rgb'
    dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/512x512rgb'
    dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/128x128rgb_white'
    dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/128x128rgb'
    #dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/256x256rgb'
    #dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair/variations/256x256rgb'
    #dataset_path='/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair'


    config_path='configs/ryan_syn.json'
 
    if train:
        launch_trainer(experiment_name,gpu_id,dataset_path,config_path)
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()
    
    checkpoint_path=get_checkpoint_path(experiment_name)

    if images:    
        render_imgs(checkpoint_path,dataset_path)
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()
    
    if video:
        video_path=render_imgs_circle(checkpoint_path,dataset_path)
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()

    return video_path




#DIFFUSION SETUP
with rp.SetCurrentDirectoryTemporarily(diffusion_model_folder):
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(diffusion_device)

    diffusion = GaussianDiffusion(
        model,
        image_size=resolution,
        timesteps=1000,  # number of steps
        sampling_timesteps=100,
        objective="pred_x0",  # We wanna use this, not noise...make my life easier lol...dont have to worry about messing the math up. Modify the model_predictions function
        loss_type="l1",  # L1 or L2
        modify_predictions=modify_predictions,
    ).to(diffusion_device)

    trainer = Trainer(
        diffusion,
        combined_photo_folder,
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

    ## load diffusion checkpoint
    if get_latest_milestone() is not None:
        print("Loading checkpoint...")
        trainer.load(get_latest_milestone())
        print("...checkpoint loaded!")

    icecream.ic(diffusion_device, dataset_path, rp.get_current_directory())


###############


# trainer.ema.ema_model.train()
# trainer.train()

with torch.no_grad():
    trainer.ema.ema_model.eval()
    i = trainer.ema.ema_model.sample(16)
