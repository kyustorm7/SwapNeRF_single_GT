
# same as eval_fid.py in cvlab4
print("running")

import sys
import os
import inspect

import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import random
from PIL import Image
import glob
import time
import random
from torchvision.utils import save_image, make_grid
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from im2scene import config
from im2scene.checkpoints import CheckpointIO
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from im2scene import config
from im2scene.checkpoints import CheckpointIO
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
logger_py = logging.getLogger(__name__)
np.random.seed(500)
torch.manual_seed(500)


# 세팅 나중에 잡기!
# random_seed = 0

# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True       # 연산속도 느려짐!
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)



# Arguments
parser = argparse.ArgumentParser(
    description='Train a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
output_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
lr = cfg['training']['learning_rate']
lr_d = cfg['training']['learning_rate_d']
batch_size = cfg['training']['batch_size']
n_workers = cfg['training']['n_workers']
t0 = time.time()

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')


dataset = config.get_dataset(cfg)
len_dset = len(dataset)
train_len = len_dset * 0.9

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(train_len), int(len_dset-train_len)])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
    pin_memory=True, drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
    pin_memory=True, drop_last=True,
)

model = config.get_model(cfg, device=device, len_dataset=len(train_dataset))


# Initialize training
op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
optimizer_kwargs = cfg['training']['optimizer_kwargs']

if hasattr(model, "generator") and model.generator is not None:
    parameters_g = model.generator.parameters()
else:
    parameters_g = list(model.decoder.parameters())
optimizer = op(parameters_g, lr=lr, **optimizer_kwargs)

if hasattr(model, "discriminator") and model.discriminator is not None:
    parameters_d = model.discriminator.parameters()
    optimizer_d = op(parameters_d, lr=lr_d)
else:
    optimizer_d = None

renderer = config.get_renderer(model, optimizer, optimizer_d, cfg, device)
checkpoint_io = CheckpointIO(output_dir, model=model, optimizer=optimizer,
                             optimizer_d=optimizer_d)
try:
    load_dict = checkpoint_io.load('model.pt')
    print("Loaded model checkpoint.")
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(output_dir, 'logs'))
logger_py.info(f'NAME: {cfg["training"]["out_dir"]} \n Major settings: \n 1) LR_G: {cfg["training"]["learning_rate"]} \n 2) LR_D: {cfg["training"]["learning_rate_d"]} \n 3) range_u: {cfg["training"]["range_u"]} \n 4) range_v: {cfg["training"]["range_v"]} \n 5) recon_weight: {cfg["training"]["recon_weight"]}')
# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)

if hasattr(model, "discriminator") and model.discriminator is not None:
    nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
    logger_py.info(
        'Total number of discriminator parameters: %d' % nparameters_d)
if hasattr(model, "generator") and model.generator is not None:
    nparameters_g = sum(p.numel() for p in model.generator.parameters())
    logger_py.info('Total number of generator parameters: %d' % nparameters_g)

t0b = time.time()



fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

out_dict_file = os.path.join(output_dir, 'fid_evaluation.npz')
out_img_file = os.path.join(output_dir, 'fid_images.npy')
out_vis_file = os.path.join(output_dir, 'fid_images.jpg')

recon_image = []
shape_swap_image = []
app_swap_image = []
pose_swap_image = []



t0 = time.time()

for batch in train_loader:
    # # Visualize output
    #if visualize_every > 0 and (it % visualize_every) == 0:
    logger_py.info('Visualizing')
    image_fake, shape_rgb, appearance_rgb, swap_rgb = renderer.visualize(batch, it=it, mode='train', val_idx=None)
    image_fake, shape_rgb, appearance_rgb, swap_rgb = image_fake.detach(), shape_rgb.detach(), appearance_rgb.detach(), swap_rgb.detach()
    for i in range(image_fake.shape[0]):
        recon_image.append(image_fake[i])
        shape_swap_image.append(shape_rgb[i])
        app_swap_image.append(appearance_rgb[i])
        pose_swap_image.append(swap_rgb[i])

img_fake = torch.stack(recon_image, dim=0)
img_fake.clamp_(0., 1.)
n_images = img_fake.shape[0]

t = time.time() - t0
out_dict = {'n_images': n_images}
out_dict['time_full'] = t
out_dict["time_image"] = t / n_images

img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
np.save(out_img_file, img_uint8)

# use unit for eval to fairy compare
img_fake = torch.from_numpy(img_uint8).float() / 255.
import pdb; pdb.set_trace()
mu, sigma = calculate_activation_statistics(img_fake)
out_dict["m"] = mu
out_dict["sigma"] = sigma

# calculate FID score and save it to a dictionary
fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
out_dict['fid'] = fid_score
print("FID Score (%d images): %.6f" % (n_images, fid_score))
np.savez(out_dict_file, **out_dict)

# Save a grid of 16x16 images for visualization
save_image(make_grid(img_fake, nrow=4, pad_value=1.), out_vis_file)