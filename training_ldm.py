
import os
os.chdir(r"Your_directory")
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Normalize
import pandas as pd
# Preprocessing

from torchvision.transforms import Compose
import math

CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -3000, "max": 3000, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        def mask_lambda_func(x):
            return limb_mask_transform(x)
        transforms.append(mask_lambda_func)
        # transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    if target_size != None:
        transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    def test_lambda_func(x):
        return lambda_transform(x)
    transforms.append(test_lambda_func)
    # transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)

def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




def save_images(img, path):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        # v = vis(images, channel_to_map(171))
        v = Image.fromarray(images)
        V.append(v)
    for value in V:
        value.save(path)

def save_tensor_as_png(tensor, filename):
    # Make sure the tensor is in the CPU and detach it from the computational graph
    tensor = tensor.detach().cpu()

    # Convert the tensor to a PIL image
    if tensor.shape[0] == 1:
        # If the input tensor has only 1 channel, convert it to a grayscale image
        image = Image.fromarray((tensor.squeeze(0).numpy() * 255).astype('uint8'), mode='L')
    else:
        # If the input tensor has 3 channels, convert it to an RGB image
        image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    # Save the image to the specified filename
    image.save(filename)


# Data loading

from torch.utils.data import Dataset
from astropy.io import fits

mapping = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}


class PairedJP2Dataset(Dataset):
    def __init__(self, dir2, labels_list, time_mag_list, transform2=None):
        super(PairedJP2Dataset, self).__init__()
        
        # Ensure directories exist
        assert os.path.isdir(dir2), f"{dir2} is not a directory."

        self.dir2_files = sorted([os.path.join(dir2, fname) for fname in os.listdir(dir2) if fname.endswith('.fits')])
        self.transform2 = transform2
        self.labels = labels_list
        self.time_mag = time_mag_list

    def __len__(self):
        return len(self.dir2_files)

    def __getitem__(self, idx):
        
        # FITS path
        
        with fits.open(self.dir2_files[idx]) as hdul:
            data2 = hdul[1].data
            data2 = np.nan_to_num(data2, nan=np.nanmin(data2))
            # data2 = data2 / 10**3
            header_1 = hdul[1].header

        data2 = to_tensor(data2)
        
        min_data2 = torch.min(data2)
        
        # Apply any transformations if provided
        if self.transform2:
            data2 = self.transform2(data2)
        label = self.labels[idx]
        time_mag = self.time_mag[idx]    
        
        return data2, label, time_mag, header_1['CDELT1'], header_1['CDELT2'], header_1['CRPIX1'], header_1['CRPIX2'], header_1['RSUN_OBS'], min_data2


dir2 = 'Your_directory'
 
df_lab = pd.read_csv("Your_directory")


import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CustomRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)


transform_hmi = get_default_transforms(
    target_size=None, channel="magnetogram", mask_limb=False, radius_scale_factor=1.0)



res_fin = transforms.Resize((256, 256))

to_tensor = transforms.ToTensor()

dataset = PairedJP2Dataset(dir2, df_lab['Label'], df_lab['MAG'], transform_hmi)


from torch.utils.data import random_split

total_samples = len(dataset)
train_size = int(0.7 * total_samples)  # Using 80% for training as an example
val_size = total_samples - train_size


torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

import random
random.seed(42)



train_data = DataLoader(train_dataset, batch_size=4,
                           shuffle=False,)
                            # pin_memory=True,# pin_memory set to True
                            # num_workers=12,
                            # prefetch_factor=4,  # pin_memory set to True
                            # drop_last=False)

val_data = DataLoader(val_dataset, batch_size=4,
                           shuffle=False,)
                            # pin_memory=True,# pin_memory set to True
                            # num_workers=12,
                            # prefetch_factor=4,  # pin_memory set to True
                            # drop_last=False)



print('Train loader and Valid loader are up!')

# Start Training

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_Conditional"
args.epochs = 500
args.batch_size = 8
args.image_size = 128
args.device = "cuda"
args.lr = 3e-4

# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import optim
import copy
import logging
import torch.nn as nn

# from utils import plot_images, save_images, get_data
from modules import PaletteModelV2, EMA
from diffusion import Diffusion_cond
from torch.utils.tensorboard import SummaryWriter

def setup_logging(run_name):
    """
    Setting up the folders for saving the model and the results

    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


setup_logging(args.run_name)
device = args.device
dataloader = train_data
dataloader_val = val_data
model = PaletteModelV2(c_in=8, c_out=4, num_classes=5,  image_size=int(128), true_img_size=128, clip_latent=None).to(device)


optimizer = optim.AdamW(model.parameters(), lr=args.lr)

mse = nn.MSELoss()
diffusion = Diffusion_cond(img_size=args.image_size, device=device, img_channel=4)
logger = SummaryWriter(os.path.join("runs", args.run_name))
l = len(dataloader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

wandb.init(project="project_name", entity="user_name")

wandb.config = {"# Epochs" : 30,
                "Batch size" : 8,
                "Image size" : 256,
                "Device" : "cuda",
                "Lr" : 3e-4}

wandb.watch(model, log=None)

min_valid_loss = np.inf

downsampl = transforms.Resize((256, 256))

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def renorm(x):
    minval = []
    maxval = []
    tensor = []
    for value in x:
        value = value.reshape(1, 1024, 1024)
        x_min = value.min()
        x_max = value.max()
        x_normalized = (value - x_min) / (x_max - x_min)
        x_scaled = 2 * x_normalized - 1
        minval.append(x_min.item())
        maxval.append(x_max.item())
        tensor.append(x_scaled.reshape(1, 1, 1024, 1024))
    
    return torch.cat(tensor, dim=0), minval, maxval

def unnorm(x_scaled, x_min, x_max):
    x_normalized = (x_scaled + 1) / 2
    x_original = x_normalized * (x_max - x_min) + x_min
    return x_original

def upscale(x):
    # Ensure x is in shape (batch_size, channels, height, width)
    # Repeat elements in the height and width dimensions
    x_repeated = x.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    
    return x_repeated

def imshow(x, y):
    true_img = x.permute(1, 2, 0).cpu().detach().numpy()
    
    flaring = y.permute(1, 2, 0).cpu().detach().numpy()
    # Create a figure with two subplots
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the original image in the first subplot
    ax1.imshow(true_img, origin='lower')
    ax1.set_title('24 before flare')
    # ax1.invert_yaxis()
    
    ax2.imshow(flaring, origin='lower')
    ax2.set_title('Flaring time')

    # ax2.invert_yaxis()

    # Show the plot
    plt.show()


def imshow_2(gt, x):
    true_img = gt[0].reshape(1, 1024, 1024).permute(1, 2, 0).cpu().numpy()
    ema_samp = x[0].permute(1, 2, 0).detach().cpu().numpy()
    # Create a figure with two subplots
    
    fig, (ax1, ax3) = plt.subplots(1, 2)

    # Plot the original image in the first subplot
    ax1.imshow(true_img, origin='lower') #, cmap=matplotlib.colormaps['hmimag'])
    ax1.set_title('Ground Truth')

    ax3.imshow(ema_samp, origin='lower') #, cmap=matplotlib.colormaps['hmimag'])
    ax3.set_title('Super-resolved Image')
    
    # Add a big title in the middle of all subplots
    fig.suptitle('Time: {} \n Label: {}'.format(time, label))
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def gamma_transform(x, c, gamma):
    s_tr = x/c
    s_tr = s_tr ** (1/gamma)
    return s_tr

def normalize_tensor(tensor):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return 2*normalized_tensor - 1
    
import gc  # Import Python's garbage collector

from diffusers import AutoencoderKL

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
enc_model = AutoencoderKL.from_single_file(url)
enc_model.to(device)

for param in enc_model.parameters():
    param.requires_grad = False

        
cropping = transforms.Compose([
    transforms.RandomCrop((1024, 1024))
])

def split_into_crops(image, crop_size):
    _, h, w = image.shape
    crops = []
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            crop = image[:, i:i+crop_size, j:j+crop_size]
            crops.append(crop)
    return torch.stack(crops)


def reassemble_crops(crops, original_size, crop_size):
    num_channels, height, width = original_size
    reassembled_image = torch.zeros(original_size)
    crop_idx = 0
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            reassembled_image[:, i:i+crop_size, j:j+crop_size] = crops[crop_idx]
            crop_idx += 1
    return reassembled_image

# Reassemble the crops back into the original image tensor
crop_size = 1024
original_size = (1, 4096, 4096)        
    


# Training loop
for epoch in range(args.epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(train_data)
    model.train()
    train_loss = 0.0
    # psnr_train = 0.0
    for i, (image_94, label, time, cdelt1, cdelt2, crpix1, crpix2, rsun_obs, mindata) in enumerate(pbar):
        
        img_24 = (image_94).to(device).float()
        
        img_24_crop = cropping(img_24)
        
        down_24 = downsampl(img_24_crop)
        up_24 = upscale(down_24.reshape(down_24.shape[0], 1, 256, 256))
        
        img_24_replicated_downscaled = up_24.repeat(1, 3, 1, 1).float().to('cuda')
        
        img_24_replicated = img_24_crop.repeat(1, 3, 1, 1).float().to('cuda')
        
        
        with autocast():
            encoded_output_24 = enc_model.encode(img_24_replicated.reshape(down_24.shape[0], 3, 1024, 1024))
            latent_representation_24 = encoded_output_24.latent_dist.sample().reshape(down_24.shape[0], 4, 128, 128)
            
            
            encoded_output_24_downscaled = enc_model.encode(img_24_replicated_downscaled.reshape(down_24.shape[0], 3, 1024, 1024))
            latent_representation_24_downscaled = encoded_output_24_downscaled.latent_dist.sample().reshape(down_24.shape[0], 4, 128, 128)
        
            residual_latent = latent_representation_24 - latent_representation_24_downscaled
        
        label = label
        time = time
        
        labels = None
        t = diffusion.sample_timesteps(residual_latent.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(residual_latent, t)
        
        
        with autocast():
            predicted_noise = model(x_t, latent_representation_24_downscaled, labels, t, None) 
            loss = mse(noise, predicted_noise)
        
        
        optimizer.zero_grad()
        
        if np.isnan(loss.item()):
            print('The batch number is: {}'.format(i))
            del img_24
            del img_peak
            del predicted_noise
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            peak_allocated = torch.cuda.max_memory_allocated(device=device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.step_ema(ema_model, model)
            peak_cached = torch.cuda.max_memory_cached(device=device)
            
            # delete unnecessary variables to save memory
            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.detach().item() * img_24.size(0)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len(pbar) + i)
    
    # Clean up memory before validation
    torch.cuda.empty_cache()
    gc.collect()

    # Validation step
    valid_loss = 0.0
    pbar_val = tqdm(val_data)
    model.eval()
    with torch.no_grad():
        for i, (image_94, label, time, cdelt1, cdelt2, crpix1, crpix2, rsun_obs, mindata) in enumerate(pbar_val):
            
            img_24 = (image_94).to(device).float()
            
            img_24_crop = cropping(img_24)
            
            down_24 = downsampl(img_24_crop)
            up_24 = upscale(down_24.reshape(down_24.shape[0], 1, 256, 256))
            
            img_24_replicated_downscaled = up_24.repeat(1, 3, 1, 1).float().to('cuda')
            
            img_24_replicated = img_24_crop.repeat(1, 3, 1, 1).float().to('cuda')
            
            
            with autocast():
                encoded_output_24 = enc_model.encode(img_24_replicated.reshape(down_24.shape[0], 3, 1024, 1024))
                latent_representation_24 = encoded_output_24.latent_dist.sample().reshape(down_24.shape[0], 4, 128, 128)
                
                
                encoded_output_24_downscaled = enc_model.encode(img_24_replicated_downscaled.reshape(down_24.shape[0], 3, 1024, 1024))
                latent_representation_24_downscaled = encoded_output_24_downscaled.latent_dist.sample().reshape(down_24.shape[0], 4, 128, 128)
            
                residual_latent = latent_representation_24 - latent_representation_24_downscaled
            
            
            
            label = label
            time = time
            
            labels = None
            t = diffusion.sample_timesteps(residual_latent.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(residual_latent, t)
            
            
            with autocast():
                predicted_noise = model(x_t, latent_representation_24_downscaled, labels, t, None) # model(x_t, None, labels, t, clip_emb_up_24)
                loss = mse(noise, predicted_noise)
            
            
            
            
            if np.isnan(loss.item()):
                del img_24
                del img_peak
                del predicted_noise
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                valid_loss += loss.detach().item() * img_24.size(0)

        # Clean up memory after validation
        torch.cuda.empty_cache()
        gc.collect()
        
    # Logging and saving
    if epoch % 5 == 0:
        ema_sampled_images = diffusion.sample(ema_model, y=latent_representation_24_downscaled[0].reshape(1, 4, 128, 128), sum_y=latent_representation_24_downscaled[0].reshape(1, 4, 128, 128), clip_enc=None, maxval=0, minval=0, labels=None, n=1)
       
        with autocast():
            
            decoded_output = enc_model.decode(latent_representation_24[0].reshape(1, 4, 128, 128))
            reconstructed_image = decoded_output.sample
            
            ema_sampled_images = ema_sampled_images + latent_representation_24_downscaled[0].reshape(1, 4, 128, 128)
            
            
            decoded_output_gen = enc_model.decode(ema_sampled_images.reshape(1, 4, 128, 128))
            reconstructed_image_gen = decoded_output_gen.sample
        
        
        
        img_rec_gt = reconstructed_image.mean(dim=1, keepdim=True)
        
        img_rec_gen = reconstructed_image_gen.mean(dim=1, keepdim=True)
        
        img_rec_gen = (img_rec_gen.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        img_rec_gen = (img_rec_gen * 255).type(torch.uint8) # to bring in valid pixel range
        
        save_images(img_rec_gen, os.path.join("results", args.run_name, f"{epoch}_ema_cond.png"))
        true_img = img_rec_gt[0].reshape(1, 1024, 1024).permute(1, 2, 0).cpu().numpy()
        ema_samp = img_rec_gen[0].permute(1, 2, 0).cpu().numpy()
        # Create a figure with two subplots
        
        fig, (ax1, ax3) = plt.subplots(1, 2)

        # Plot the original image in the first subplot
        ax1.imshow(true_img, origin='lower')
        ax1.set_title('24 before flare')

        ax3.imshow(ema_samp, origin='lower')
        ax3.set_title('Predicted flaring image')
        
        # Add a big title in the middle of all subplots
        fig.suptitle('Time: {} \n Label: {}'.format(time[0], label[0]))
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
        

    wandb.log({
        "Training Loss": train_loss / len(train_data),
        "Validation Loss": valid_loss / len(val_data),
        'Sampled images': wandb.Image(plt)
    })

    plt.close()
    
    if min_valid_loss > valid_loss:
        logging.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
    
    # Saving State Dict
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_test_cond_epoch{epoch}.pt"))
    torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_cond_epoch{epoch}.pt"))
    state = {
        'model_state': ema_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, os.path.join("models", args.run_name, "checkpoint.pt"))
    wandb.save('ema_model_epoch{}.pt'.format(epoch))
    wandb.save('model_epoch{}.pt'.format(epoch))
            
            
            
            
            