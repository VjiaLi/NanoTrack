import numpy as np
import torch
import cv2
import os
from ultralytics.utils import ops
from pathlib import Path

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def write_mot_results(txt_path, results, frame_idx):
    results = torch.from_numpy(results)

    if results.numel() == 0:
        results = results.reshape(-1, 7)

    nr_dets = len(results)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)

    mot = torch.cat([
        frame_idx,
        results[:, 4].unsqueeze(1).to('cpu'),
        ops.xyxy2ltwh(results[:, 0:4]).to('cpu'),
        results[:, 5].unsqueeze(1).to('cpu'),
        results[:, 6].unsqueeze(1).to('cpu'),
        dont_care
    ], dim=1)

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # create mot txt file
    txt_path.touch(exist_ok=True)

    with open(str(txt_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%d')  # save as ints instead of scientific notation


def create_exp(args):
    if not os.path.exists(Path(args.project)):
        os.makedirs(Path(args.project))

    exp_files = []
    for filename in os.listdir(str(args.project)):
        if filename.startswith("exp") and filename[3:].isdigit():
            exp_files.append(filename)

    if not exp_files:
        max_num = 0
    else:      
        max_num = max(int(filename[3:]) for filename in exp_files)

    if not os.path.exists(str(args.project / args.name) + f"{max_num+1}"):
        os.makedirs(str(args.project / args.name) + f"{max_num+1}")

    dir = str(args.project / args.name) + f"{max_num+1}"
    video_filename = os.path.join(dir, "result.avi")
    out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'XVID'),30,(1920,1080))
    return out

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names