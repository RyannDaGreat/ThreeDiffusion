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

#GO INTO THE RIGHT DIRECTORY
rp.set_current_directory('/raid/ryan/CleanCode/Projects/Experimental/diffusion_for_nerf/source')

#SETUP ASSERTIONS
assert rp.get_current_directory().endswith('/diffusion_for_nerf/source')

#PATH SETTINGS
project_root='..' #We're in the 'source' folder of the project

#lego mode
dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/lego"
diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_lego_direct_128')

#hotdog mode
dataset_path = "/home/ryan/CleanCode/Datasets/nerf/nerf_synthetic/hotdog"
diffusion_model_folder = rp.path_join(project_root,'diffusion_models/diffusion_hotdog_direct')


plenoxel_opt_folder = '/raid/ryan/CleanCode/Github/svox2/opt' #TODO: Move this into the project
plenoxel_experiment_name = 'sandbox_for_plenoxel_diffusion'

#OTHER SETTINGS
NUM_ITER=999 #Between 1 and 999. 10 is not enough.
NUM_FIXED=5 #Number of fixed ground truth images.

#DIFFUSION SETTINGS
diffusion_device = torch.device("cuda:1")
plenoxel_device = torch.device("cuda:0")
resolution=128 #Later on, this should be detected from the diffusion_model_folder

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
image_filenames=[rp.get_file_name(frame['file_path']) for frame in dataset_json['frames']]
image_filenames=[rp.with_file_extension(x,'png',replace=True) for x in image_filenames]

#MAX BATCH SIZE FOR RESOLUTIONS:
#    128x128 : 225
#    256x256 : ?
BATCH_SIZE=len(dataset_json['frames'])
icecream.ic(BATCH_SIZE)

#SETTINGS VALIDATION
assert 0<=NUM_FIXED<=BATCH_SIZE

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
print("JM")
with SetCurrentDirectoryTemporarily(trading_folder):
    print("J))M")
    rp.fansi_print("TRADING FOLDER: " + rp.get_current_directory(), "cyan", "bold")

    #We're not going to bother with test or validation sets: only training.
    rp.copy_file(training_json_file,'transforms_train.json')
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
    image_files=rp.get_all_image_files(images_folder,sort_by='number')
    assert len(image_files)==len(image_filenames),'Mismatch! Whats going on here? We want %i images but found %i in %s'%(len(image_files),len(image_filenames),images_folder)
    rp.fansi_print("LOADING IMAGES FROM PLENOXEL OUTPUT FOLDER...",'cyan','bold')
    images=rp.load_images(image_files,show_progress=True)
    rp.fansi_print("...DONE!",'cyan','bold')

    #The plenoxel code outputs images that compare ground truth to its own output, side by side. We only want the output, so we take the right half of each image.
    images=[rp.split_tensor_into_regions(x,1,2)[1] for x in images]

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
    fixed_images=load_images(image_filenames,show_progress=True)
    fixed_images=[rp.cv_resize_image(x,(resolution,)*2) for x in fixed_images]


ITER=0
def modify_predictions(images):
    # images = [modify_prediction(image) for image in images]

    global ITER

    print('\n\n\n\n================  STARTING ITER %i  /  %i  ===============\n\n'%(ITER,NUM_ITER))

    ITER+=1

    rp.fansi_print("MODIFYING PREDICTIONS!",'cyan','bold')
    before_image=(rp.labeled_image(rp.tiled_images(images),'Diffusion Output',size=50)) #Display images before...

    save_images_to_trade(images)  # Where plenoxels can see them...
    output_images_folder = test_launch_trainer(train=True, images=True, video=False)
    images = load_plenoxel_renderings(output_images_folder)

    after_image=rp.labeled_image(rp.tiled_images(images),'Plenoxel Output',size=50) #Display after-images...

    images[:NUM_FIXED]=fixed_images[:NUM_FIXED]

    display_image_on_macmax(
        rp.labeled_image(
            rp.tiled_images(
                [before_image, after_image],
                border_color=(1, 0, 0, 1),
                border_thickness=10,
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
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(diffusion_device)

    diffusion = GaussianDiffusion(
        model,
        image_size=resolution,
        timesteps=1000,  # number of steps
        # sampling_timesteps=100,
        sampling_timesteps=NUM_ITER,
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
    i = trainer.ema.ema_model.sample(BATCH_SIZE)
    test_launch_trainer(train=False,images=False,video=True)
    print("Don't forget to use FRAMES")
