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

class EvalImageDataset(torch.utils.data.Dataset):
    """
    Get Images for Evaluation
    """
    def __init__(
            self, path, image_size=(64, 64),
    ):
        super().__init__()
        self.image_size = image_size
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        import time
        t0 = time.time()
        print("Start loading file addresses ...")

        # image_filetype = ('*.png', '*.jpg')
        self.images = []
        valid_images = ["*.jpg", "*.gif", "*.png"]
        if os.path.exists(self.path):
            for valid_image_filename in valid_images:
                print(self.path)
                self.images.extend(glob.glob(os.path.join(self.path, valid_image_filename)))
        else:
            raise OSError("No Path Exists")

        #random.shuffle(images)
        load_t = time.time() - t0
        print('Finished loading file addresses, time:', load_t)
        print("Number of images found: %d" % len(self.images))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        data = {
            'path': os.path.basename(image_path),
            'image': image
        }
        return data

model_dir = args.model.strip()          
output_dir = args.output.strip()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out_dict_file = os.path.join(output_dir, 'fid_evaluation.npz')
out_img_file = os.path.join(output_dir, 'fid_images.npy')
out_vis_file = os.path.join(output_dir, 'fid_images.jpg')

eval_dset = EvalImageDataset(args.datadir)
eval_loader = torch.utils.data.DataLoader(
    eval_dset, batch_size = 2, shuffle=True, num_workers = 8, pin_memory=True
)


# Load Trained Model
model = config.get_model(cfg, device=device)

checkpoint_io = CheckpointIO(model_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])
print("Loaded Model from", cfg['test']['model_file'])

# Generate
renderer = config.get_renderer(model, cfg, device=device)
model.eval()

fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])

batch_size = 2
n_iter = 3
n_images = batch_size * 4 * n_iter
out_dict = {'n_images': n_images}

img_fake = []
t0 = time.time()
iter_counter = 0

with torch.no_grad():
    for batch in eval_loader:
        print("Generating Images for", batch["path"][0], "and", batch["path"][1])
        out = renderer.render_full_visualization(batch['image'])
        img_fake.extend(out.values())
        """
        for i in range(len(batch)):
            recon_path = os.path.join(output_dir, "recon_" + batch["path"][i] )
            save_image(out["recon"][i].detach(), recon_path )
            shape_swap_path = os.path.join(output_dir, "shape_swapped_" + batch["path"][i] )
            save_image(out["shape"][i].detach(), shape_swap_path )
            appearance_swap_path = os.path.join(output_dir, "appearance_swapped_" + batch["path"][i] )
            save_image(out["appearance"][i].detach(), appearance_swap_path)
            pose_swap_path = os.path.join(output_dir, "pose_swapped_" + batch["path"][i] )
            save_image(out["pose"][i].detach(), pose_swap_path)

        """
        iter_counter += 1
        if iter_counter <= n_iter:
            break

img_fake = torch.cat(img_fake, dim=0)[:n_images]
img_fake.clamp_(0., 1.)
n_images = img_fake.shape[0]

t = time.time() - t0
out_dict['time_full'] = t
out_dict["time_image"] = t / n_images

img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
np.save(out_img_file[:n_images], img_uint8)

# use uint for eval to fairy compare
img_fake = torch.from_numpy(img_uint8).float() / 255.
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



    
        