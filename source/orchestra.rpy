#SUMMARY
#    - We use the blender datasets provided by Plenoxel.
#    - We train on their test data, and test on their train data:
#          We train the diffusion model on the test images, because there are more of them.
#          We train the plenoxel model with the positions in the test data, because there are 200 of them
#          There are 200 images in the train datasets, and 600 in the test
#    - We train our diffusion model with 128x128 images, and have a batch size of 200 (the number of training samples)
#    - This process does the diffusion, and coordinates with the plenoxel model code by running it in separate processes.
# Diffusion Code from https://github.com/lucidrains/denoising-diffusion-pytorch


#THOUGHT PROCESS:
#  HINTS: It seems to need something to disambiguate...
#  HINT_REPEAT: Large batch sizes seem to ignore the hint, drowning it out...
#  Gradual Expansion: Just a hunch, like the 'fibbing' idea...what if we just paint outwards?
#      (maybe starting off with a hint and a view that can't see any of that hint is not only useless but actively bad?)
#  OVERTIME: It seems to be making so much progress...but all at the last minute lol.
#      For sum reason, increasing NUM_ITER doesn't always help...actually it's never helped lol. Something about the earlier ITER's thows it off, idk why...but repeating the last diffusion step over and over seems to help a LOT LOT LOT.
#  sh_dim=0: No holograms! Holograms are bad lol
#TODO: Make variable scheduled batch sizes; gradually increase spatial resolution once cameras are aligned (from 16x16 upward to prevent 2-view overfitting)


#IMPORTS
import rp
import os
import rp.web_evaluator as web_evaluator
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import icecream
import inspect
import random
import time

#GO INTO THE RIGHT DIRECTORY
rp.set_current_directory('/raid/ryan/CleanCode/Projects/Experimental/diffusion_for_nerf/source')

#SETUP ASSERTIONS
assert rp.get_current_directory().endswith('/diffusion_for_nerf/source')

#PATH SETTINGS
project_root='..' #We're in the 'source' folder of the project

##drums mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/drums"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_drums_direct_128')
#resolution=128 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8)

##chair mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/chair"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_chair_direct_128')
#resolution=128 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8)

#lego mode
dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego"
diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_lego_direct_128')
resolution=128 #Later on, this should be detected from the diffusion_model_folder
DIM_MULTS=(1, 2, 4, 8)

##hotdog mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/hotdog"
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/drums"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_hotdog_direct_128')
#resolution=128 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8)

##drums256 mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/drums"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_drums_direct_256')
#resolution=256 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8, 16)

##lego256 mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_lego_direct_256')
#resolution=256 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8, 16)

##ficus256 mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/ficus"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_ficus_direct_256')
#resolution=256 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8, 16)

##hotdog256 mode
#dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/hotdog"
#diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_hotdog_direct_256')
#resolution=256 #Later on, this should be detected from the diffusion_model_folder
#DIM_MULTS=(1, 2, 4, 8, 16)









plenoxel_opt_folder = '/raid/ryan/CleanCode/Github/svox2/opt' #TODO: Move this into the project
plenoxel_experiment_name = 'sandbox_for_plenoxel_diffusion'

##OTHER SETTINGS
#NUM_ITER=20 #Between 1 and 999. 10 is not enough.
#NUM_HINTS=3 #Number of fixed ground truth images.
#HINT_REPEAT=4 #Number of times we repeat the hints, to give them more weight...total number is NUM_HINTS*HINT_REPEAT, and that takes away from BATCH_SIZE
#BATCH_SIZE=20 #Can be None indicating to use the whole training set, or an int overriding it
#BATCH_SIZE+=NUM_HINTS * HINT_REPEAT

#OTHER SETTINGS

#PHASE 1
NUM_ITER=5 #Between 1 and 999. 10 is not enough.
NUM_ITER=2
OVERTIME=100 #Repeat the last timestep this number of times. It seems to make a lot of progress at the last minute.
NUM_HINTS=1 #Number of fixed ground truth images.
HINT_REPEAT=8 #Number of times we repeat the hints, to give them more weight...total number is NUM_HINTS*HINT_REPEAT, and that takes away from BATCH_SIZE
# BATCH_SIZE=2 #Can be None indicating to use the whole training set, or an int overriding it. If it's too large it might not work as well, as the camera distribution is no longer uniform.
BATCH_SIZE=17
BATCH_SIZE+=NUM_HINTS * HINT_REPEAT
SHUFFLE_CAMERAS=True #If True, we shuffle the camera positions in the dataset - making hints unable to give correct positions. Used to test robustness, but will probably give worse results

SEED=time.time_ns()
# SEED=1663104616704513119
SEED=1663108221093693582
random.seed(SEED)

# BATCH_SIZE=None

#DIFFUSION SETTINGS
diffusion_device = torch.device("cuda:0") #If you run out of vram, set these to two different devices
plenoxel_device = torch.device("cuda:1")

#DERIVED PATHS:
combined_photo_folder = rp.path_join(dataset_path,'combined_photos') #For training the diffusion model
trading_folder=rp.make_directory(rp.temporary_file_path()) #Will be in /tmp/local or something...ideally not on /raid
training_json_file = rp.path_join(dataset_path,'transforms_train.json')

#PATH SETTINGS VALIDATION
assert rp.folder_exists(project_root)
assert rp.folder_exists(combined_photo_folder)
assert rp.folder_exists(diffusion_model_folder)
assert rp.folder_exists(project_root)
assert rp.folder_exists(dataset_path)
assert rp.folder_exists(plenoxel_opt_folder)
assert rp.folder_exists(trading_folder)
assert rp.file_exists(training_json_file)

#FINALIZE PATH SETTINGS
project_root           = rp.get_absolute_path(project_root          )
combined_photo_folder  = rp.get_absolute_path(combined_photo_folder )
diffusion_model_folder = rp.get_absolute_path(diffusion_model_folder)
project_root           = rp.get_absolute_path(project_root          )
dataset_path           = rp.get_absolute_path(dataset_path          )
plenoxel_opt_folder    = rp.get_absolute_path(plenoxel_opt_folder   )
trading_folder         = rp.get_absolute_path(trading_folder        )
training_json_file     = rp.get_absolute_path(training_json_file    )

#SETUP
torch.cuda.set_device(diffusion_device)
dataset_json=rp.load_json(training_json_file)

#This is why we need to know the seed:
dataset_json['frames']=rp.shuffled(dataset_json['frames'])

if SHUFFLE_CAMERAS: 
    rp.fansi_print("SHUFFLING CAMERAS!!!",'yellow','bold')
    camera_transforms=[x['transform_matrix'] for x in dataset_json['frames']]
    camera_transforms=rp.shuffled(camera_transforms)
    for frame,transform in zip(dataset_json['frames'],camera_transforms):
        frame['transform_matrix']=transform

dataset_json['frames']=dataset_json['frames'][:NUM_HINTS]*HINT_REPEAT + dataset_json['frames'][NUM_HINTS:]

#Only use cameras similar to the hints...
#This is my hypothesis: If we start from a completely unseen view, it won't work. We need to expand from a starting view and slowly go outward...
#We'll call this "Gradual Expanson" for now. Gradual because it will probably learn the easy ones first, then solve the hard ones...
def frame_dist(frame):
    #How close is a frame to a hint?
    target_frames=dataset_json['frames'][:NUM_HINTS]
    x=frame['transform_matrix']
    targets=[target_frame['transform_matrix'] for target_frame in target_frames]
    dists=[rp.euclidean_distance(x,y) for y in targets]
    return min(dists)

dataset_json["frames"][NUM_HINTS * HINT_REPEAT :] = sorted(
    dataset_json["frames"][NUM_HINTS * HINT_REPEAT :], key=frame_dist
)[::1]


if BATCH_SIZE is not None:
    #For artificially limiting the batch size
    dataset_json['frames']=dataset_json['frames'][:BATCH_SIZE]
else:
    BATCH_SIZE=len(dataset_json['frames'])
image_filenames=[rp.get_file_name(frame['file_path']) for frame in dataset_json['frames']]
image_filenames=[rp.with_file_extension(x,'png',replace=True) for x in image_filenames]


#SETTINGS VALIDATION
assert 0<=NUM_HINTS*HINT_REPEAT<=BATCH_SIZE
#MAX BATCH SIZE FOR RESOLUTIONS:
#    128x128 : 225
#    256x256 : ?
assert BATCH_SIZE==len(dataset_json['frames'])
icecream.ic(SEED,BATCH_SIZE,NUM_HINTS,HINT_REPEAT)

#GENERAL HELPER FUNCTIONS
@rp.memoized
def get_macmax_client():
    client=web_evaluator.Client('172.27.78.46')
    return client

if 'FRAMES' not in dir():
    FRAMES=[]
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
            "x=decode_image_from_bytes(x);display_image(x);save_image(x);frames.append(x);text_to_speech('dingo')",
            x=rp.encode_image_to_bytes(image),
        )
        FRAMES.append(image)
    except Exception as e:
        rp.fansi_print('WebEval Failed: '+str(e),'red')

def tiled_torch_images(images):
    images = rp.as_numpy_images(images)
    images = rp.tiled_images(images)
    return images

class SetCurrentDirectoryTemporarily:
    #Temporarily CD into a directory
    #Modified from rp
    def __init__(self,directory):
        self.directory=directory
        
    def __enter__(self):
        self.original_dir=rp.get_current_directory()
        if self.directory is not None:
            rp.set_current_directory(self.directory)
        rp.fansi_print("ENTERING DIR: "+rp.get_current_directory(),'yellow')
            
    def __exit__(self,*args):
        rp.set_current_directory(self.original_dir)
        rp.fansi_print("EXITING DIR, NOW IN "+rp.get_current_directory(),'yellow')

def do_in_dir(folder):
    #Decorator for functions - uses SetCurrentDirectoryTemporarily
    def decorator(func):
        def wrapper(*args,**kwargs):
            with SetCurrentDirectoryTemporarily(folder):
                return func(*args,**kwargs)
        wrapper.signature=inspect.signature(func) #Better rp autocompletion
        return wrapper
    return decorator


#ORCHESTRATOR
with SetCurrentDirectoryTemporarily(trading_folder):
    rp.fansi_print("TRADING FOLDER: " + rp.get_current_directory(), "cyan", "bold")

    #We're not going to bother with test or validation sets: only training.
    rp.save_json(dataset_json,'transforms_train.json',pretty=True)
    rp.make_symlink('transforms_train.json','transforms_test.json') #This might not be necessary
    rp.make_symlink('transforms_train.json','transforms_val.json') #This might not be necessary

    rp.make_directory('train')
    rp.make_symlink('test','train')
    rp.make_symlink('val','train')

def save_images_to_trade(images):
    assert len(images)==len(image_filenames), 'All images should be in the same length and order as the image filenames'
    with SetCurrentDirectoryTemporarily(trading_folder):
        rp.set_current_directory('train')
        rp.fansi_print("SAVING IMAGES TO TRADE FOLDER...",'cyan','bold')
        rp.save_images(images,image_filenames,show_progress=True)
        rp.fansi_print("...DONE!",'cyan','bold')

# def load_images_from_trade():
#     with SetCurrentDirectoryTemporarily(trading_folder):
#         rp.set_current_directory('train')
#         rp.fansi_print("LOADING IMAGES FROM TRADE FOLDER...",'cyan','bold')
#         output=rp.load_images(image_filenames,show_progress=True)
#         rp.fansi_print("...DONE!",'cyan','bold')
#     return output

def load_plenoxel_renderings(images_folder):
    image_files=rp.get_all_image_files(images_folder,sort_by='number')[:BATCH_SIZE]
    assert len(image_files)==BATCH_SIZE,'Mismatch! Whats going on here? We want %i images but found %i in %s'%(BATCH_SIZE,len(image_files),images_folder)
    rp.fansi_print("LOADING IMAGES FROM PLENOXEL OUTPUT FOLDER...",'cyan','bold')
    images=rp.load_images(image_files,show_progress=True)
    rp.fansi_print("...DONE!",'cyan','bold')

    #The plenoxel code outputs images that compare ground truth to its own output, side by side. We only want the output, so we take the right half of each image.
    images=[rp.split_tensor_into_regions(x,1,2)[1] for x in images]
    images=[rp.as_float_image(x) for x in images]
    images=[rp.as_rgb_image(x) for x in images]

    return images



#DIFFUSION FUNCTIONS

#times=70*BATCH_SIZE
#def modify_prediction(image):
#    global times
#    if times:
#        print(times)
#        times-=1
#        #image = rotate_image(image, randint(360))
#        #image = rotate_image(image, 0)
#        #image = rotate_image(image, 30)
#    image = rp.crop_image(image, resolution, resolution, origin="center")
#    return image

print("Loading fixed images from ground truth dataset...")
with SetCurrentDirectoryTemporarily(rp.path_join(dataset_path,'train')):
    fixed_images=rp.load_images(image_filenames,show_progress=True)
    fixed_images=[rp.as_float_image(rp.as_rgb_image(rp.cv_resize_image(x,(resolution,)*2))) for x in fixed_images]


ITER=0
UNHINT_ITER=20 #After this iter, don't use the hints any more - and try to let the cameras find their way...
def modify_predictions(images):
    # images = [modify_prediction(image) for image in images]

    global ITER, UNHINT

    print('\n\n\n\n================  STARTING ITER %i  /  %i  ===============\n\n'%(ITER,NUM_ITER))

    ITER+=1
    
    if NUM_HINTS*HINT_REPEAT and not ITER>UNHINT_ITER:
        images[:NUM_HINTS*HINT_REPEAT]=fixed_images[:NUM_HINTS*HINT_REPEAT]
    elif ITER>UNHINT_ITER:
        print(rp.fansi("UNHINTING!!",'cyan','bold','blue'))

    rp.fansi_print("MODIFYING PREDICTIONS!",'cyan','bold')
    before_image=(rp.labeled_image(rp.tiled_images(images),'Diffusion Output',size=50)) #Display images before...

    save_images_to_trade(images)  # Where plenoxels can see them...
    output_images_folder = test_launch_trainer(train=True, images=True, video=False)
    loaded_images = load_plenoxel_renderings(output_images_folder)
    # images[1:]=loaded_images[1:]#Let one evolve without nerf
    images=loaded_images


    #DISPLAY
    after_image=rp.labeled_image(rp.tiled_images(images),'Plenoxel Output',size=50) #Display after-images...
    ground_truth_image=rp.labeled_image(rp.tiled_images(fixed_images[:len(images)]),'Ground Truth',size=50) #Display after-images...
    displayed_images=[before_image, after_image]
    if not SHUFFLE_CAMERAS:
        displayed_images+=[ground_truth_image]
    display_image_on_macmax(
        rp.labeled_image(
            rp.tiled_images(
                displayed_images,
                border_color=(0, 1, 0, 1) if ITER>UNHINT_ITER else (1, 0, 0, 1),
                border_thickness=10,
                length=3,
            ),
            "ITER %i / %i"% (ITER,NUM_ITER),
            size=100,
        )
    )

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
    output=rp.get_absolute_path(output)
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
    # assert rp.path_exists(out_path), "Something went wrong - the video wasnt written"   <--- This assertion is wrong!!! It triggers even when the video IS written.s
    
@do_in_dir(plenoxel_opt_folder)
def render_imgs(checkpoint_path, dataset_path):
    assert_right_plenoxel_conditions()

    #The --train flag means we evaluate the training folder. This is what we want to do.
    #Not that it would (currently) make a difference - in the diffusion plenoxel test, the test and val folders and jsons are symlinks to the train ones

    command = "python render_imgs.py --blackbg --no_vid --no_lpips --train %s %s"
    command %= (
        checkpoint_path,
        dataset_path,
    )

    rp.fansi_print("COMMAND: " + command, "green", "bold")

    os.system(command)

    output_images_folder_name='train_renders_blackbg' #This might change if we don't use --blackbg!! Also, if we don't use --train it's test_renders_blackbg
    output_images_folder = rp.path_join(rp.get_parent_folder(checkpoint_path), output_images_folder_name)
    output_images_folder = rp.get_absolute_path(output_images_folder) #It might already be absolute idk lol too lazy to check
    assert rp.folder_exists(output_images_folder)

    return output_images_folder

    
@do_in_dir(plenoxel_opt_folder)
def test_launch_trainer(train=True,images=True,video=True):
    rp.tic()
    
    experiment_name = plenoxel_experiment_name
    
    gpu_id=0
    
    dataset_path = trading_folder

    config_path='configs/ryan_syn.json'

    #FOR TESTING PLENOXEL SETTINGS:
    # experiment_name='DINGO'
    # dataset_path = '/raid/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego/variations/128x128rgb' #Can it work on current settings?
 
    if train:
        launch_trainer(experiment_name,gpu_id,dataset_path,config_path)
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()
    
    checkpoint_path=get_checkpoint_path(experiment_name)

    if images:    
        output_images_folder = render_imgs(checkpoint_path,dataset_path)
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()
    else:
        output_images_folder = None
    
    if video:
        video_path=render_imgs_circle(checkpoint_path,dataset_path)
        rp.fansi_print("CREATED VIDEO: " + video_path, None, "bold")
        rp.fansi_print(rp.toc(),'cyan','bold');rp.tic()

    return output_images_folder




#DIFFUSION SETUP
with SetCurrentDirectoryTemporarily(diffusion_model_folder):
    model = Unet(dim=64, dim_mults=DIM_MULTS).to(diffusion_device)

    diffusion = GaussianDiffusion(
        model,
        image_size=resolution,
        timesteps=1000,  # number of steps
        # sampling_timesteps=100,
        sampling_timesteps=NUM_ITER+OVERTIME,
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

if OVERTIME:
    overtime_timesteps=int(OVERTIME*(diffusion.num_timesteps/NUM_ITER))
    extra_betas=[diffusion.betas[-1]]*overtime_timesteps
    extra_betas=torch.tensor(extra_betas)
    extra_betas=extra_betas.to(diffusion_device)
    diffusion.betas=torch.cat((diffusion.betas,extra_betas))
    diffusion.num_timesteps+=overtime_timesteps




###############


# trainer.ema.ema_model.train()
# trainer.train()

with torch.no_grad():
    trainer.ema.ema_model.eval()
    i = trainer.ema.ema_model.sample(BATCH_SIZE)
    test_launch_trainer(train=False,images=False,video=True)
    print("Don't forget to use FRAMES")
