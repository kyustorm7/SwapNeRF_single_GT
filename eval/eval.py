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
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument(
    "--datadir", "-D", type=str, default=None, help="Dataset directory"
    )
parser.add_argument(
    '--split',
    type=str,
    default='test',
    help = 'Split of data to use train | val | test',
    )
parser.add_argument(
    '--output',
    '-O',
    type=str,
    default = "eval_out",
    help = "Path to output"
)
parser.add_argument(
    "--dataset_format",
    "-F",
    type=str,
    default="srn",
    help="Dataset format, multi_obj | dvr | dvr_gen | dvr_dtu | srn",
)

args = parser.parse_args()

# Load Images and Poses
class SRNPoseDataset(torch.utils.data.Dataset):
    """[summary]
    Dataset from SRN
    Get Random Pose+Image, and every Pose from one object
    """
    def __init__(
        self, path, stage="test", image_size=(128, 128), world_scale=1.0, 
    ):
        super().__init__()
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp
        
        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

    def _len_(self):
        return len(self.intrins)
    
    def _getitem_(self, index): 
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        base_path = os.path.basename(os.path.basename(dir_path))
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        total_len = len(rgb_paths)
        print("found", total_len, "viewpoints for object", base_path )
        """
        img_idx = random.randrange(total_len)
        
        input_img = Image.open(rgb_paths[img_idx]).convert("RGB")
        input_img = self.transform(input_img)
        input_pose = torch.from_numpy(
            np.loadtxt(pose_paths)
        )
        input_pose = input_pose @ self._coord_trans
        """
        all_images = []
        all_poses = []

        for idx in range(total_len):
            new_img = Image.open(rgb_paths[idx]).convert("RGB")
            new_img = self.transform(new_img)
            all_images.append(new_img)

            new_pose = torch.from_numpy(
                np.loadtxt(pose_paths[idx], dtype=np.float32.reshape(4, 4))
            )
            new_pose = new_pose @ self._coord_trans
            all_poses.append(new_pose)

        data = {
            'name': base_path,
            'n_views': total_len,
            'paths': rgb_paths,
            'images': all_images,
            'pose': all_poses,
        }
        return data

#####################################################################################

if args.dataset_format == "srn":
    dset_class = SRNPoseDataset
else:
    raise NotImplementedError("Unsupported Dataset Type")

dset = dset_class(args.dataset_format, args.split)
data_loader = torch.utils.data.DataLoader(
    dset, batch_size=1,shuffle=False, num_workers=8, pin_memory=False 
)

output_dir = args.output.strip()
#total_psnr = 0.0
#total_ssim = 0.0
cnt = 0

finish_path = os.path.join(output_dir, "finish.txt")
os.makedirs(output_dir, exist_ok=True)
if os.path.exists(finish_path):
    with open(finish_path, "r") as f:
        lines = [x.strip().split() for x in f.readlines()]
    lines = [x for x in lines if len(x) == 4]
    finished = set([x[0] for x in lines])
    total_psnr = sum((float(x[1]) for x in lines))
    total_ssim = sum((float(x[2]) for x in lines))
    cnt = sum((int(x[3]) for x in lines))
    if cnt > 0:
        print("resume psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
    else:
        total_psnr = 0.0
        total_ssim = 0.0
else:
    finished = set()

finish_file = open(finish_path, "a", buffering=1)
print("Writing images to", output_dir)


### Load Trained Model
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print("Using Device:", device)

out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
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

model = config.get_model(cfg, device=device, len_dataset=dset._len_)
checkpoint_io = CheckpointIO(out_dir, model=model)     # Load Model Checkpoint
checkpoint_io.load(cfg['test']['model_file'])

optimizer = None
optimizer_d = None

# Renderer
renderer = config.get_renderer(model, optimizer, optimizer_d, cfg, device=device)

# Render Images
with torch.no_grad():
    for batch in data_loader:
        object_name = batch['name']
        num_views = batch["n_views"]
        img_paths = batch["paths"]
        image_list = batch["images"]
        pose_list = batch["pose"]

        print("rendering images for object", object_name)
        one_object_time_start = time.time()

        # Get base image
        base_idx = random.randrange(num_views)
        base_img_path_name = os.path.basename(img_paths[base_idx])
        base_viewpoint = os.path.splitext(base_img_path_name)[0]
        print("rendering viewpoints using fixed viewpoint", base_viewpoint )
        

        img = image_list[base_idx]
        pose = pose_list[base_idx]

        object_path = os.path.join(output_dir, object_name)
        os.makedirs(object_path)
        


        for i in range(num_views):
            one_view_time_start = time.time()
            img_path_name = os.path.basename(img_paths[i])
            print("rendering viewpoint", os.path.splitext(img_path_name)[0], "using viewpoint", base_viewpoint)

            output_pose = pose_list[base_idx]
            output_image = renderer.visualize(img, pose, output_pose)
            save_image(output_image, os.path.join(object_path, img_path_name))

            






