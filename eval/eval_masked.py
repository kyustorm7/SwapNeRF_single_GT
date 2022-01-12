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
from torchvision.utils import save_image
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from im2scene import config
from im2scene.checkpoints import CheckpointIO

# Arguments
parser = argparse.ArgumentParser(
    description='Render 3D ShapeNet images from trained SwapNerf Models'
)
parser.add_argument('--config', type=str, default="configs/default.yaml",  help='Path to config file.')
parser.add_argument(
    "--datadir", "-D", type=str, default=None, help="Dataset directory"
    )

parser.add_argument(
    '--output',
    '-O',
    type=str,
    default = "eval_out",
    help = "Path to output"
)
parser.add_argument(
    '--model',
    '-M',
    type=str,
    default = "out",
    help = "Path to trained model"
)

parser.add_argument('--no-cuda', action='store_true', help="Do not use cuda")

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
if is_cuda:
    print("USING CUDA")
    device = torch.device("cuda")
else:
    print("WARNING: Not using cuda")
    device = torch.device("cpu")


model_dir = args.model.strip()          
output_dir = args.output.strip()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

eval_dataset = config.get_dataset(cfg)    
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size = 2, shuffle=True, num_workers = 8, pin_memory=True
)



# Load Trained Model
model = config.get_model(cfg, device=device)

checkpoint_io = CheckpointIO(model_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])
print("Loaded Model from", cfg['test']['model_file'])

# Generator
renderer = config.get_renderer(model, cfg, device=device)
model.eval()

with torch.no_grad():
    for batch in eval_loader:
        print("Generating Images for", batch["path"][0], "and", batch["path"][1])
        out = renderer.render_full_visualization(batch['image'])

        for i in range(len(batch)):
            import pdb
            pdb.set_trace()
            recon_path = os.path.join(output_dir, "recon_" + batch["path"][i] )
            save_image(out["recon"][i].detach(), recon_path )
            shape_swap_path = os.path.join(output_dir, "shape_swapped_" + batch["path"][i] )
            save_image(out["shape"][i].detach(), shape_swap_path )
            appearance_swap_path = os.path.join(output_dir, "appearance_swapped_" + batch["path"][i] )
            save_image(out["appearance"][i].detach(), appearance_swap_path)
            pose_swap_path = os.path.join(output_dir, "pose_swapped_" + batch["path"][i] )
            save_image(out["pose"][i].detach(), pose_swap_path)

        break
    
        