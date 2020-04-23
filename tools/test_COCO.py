import os, csv, png
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from vnl.lib.utils.net_tools import load_ckpt
from vnl.lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from tools.parse_arg_test import TestOptions
from vnl.data.load_dataset import CustomerDataLoader
from vnl.lib.models.metric_depth_model import MetricDepthModel
from vnl.lib.core.config import cfg, merge_cfg_from_file
from vnl.lib.models.image_transfer import bins_to_depth

logger = setup_logging(__name__)


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    # load model
    model = MetricDepthModel()
    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    out_dir = os.path.join(test_args.dataroot, 'VNL_Monocular')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_path = test_args.dataroot + "path_list.txt"
    with open(file_path, "r") as file:
        for path in tqdm(file, total=get_num_lines(file_path)):
            img_path = path.strip()

            out_path = os.path.join(test_args.dataroot, 'VNL_Monocular', os.path.split(path)[1])
            out_path = os.path.splitext(out_path)[0] + ".png"

            # if not os.path.exists(out_path):
            with torch.no_grad():
                try:
                    img = Image.open(img_path)
                    img_resize = img.copy()
                    img_torch = scale_torch(img_resize, 255)
                    img_torch = img_torch[None, :, :, :].cuda()

                    _, pred_depth_softmax = model.module.depth_model(img_torch)
                    pred_depth = bins_to_depth(pred_depth_softmax)

                    # Un-normalize using factor from vnl/data/nyudv2_dataset.py
                    pred_depth = 10.0 * pred_depth.cpu().numpy().squeeze()

                    # Convert to uint16 for storage
                    pred_depth_scaled = (pred_depth * 256).astype('uint16')

                    # Write out as uint16 png
                    with open(out_path, 'wb') as f:
                        writer = png.Writer(width=pred_depth.shape[1], height=pred_depth.shape[0], bitdepth=16,
                                            greyscale=True)
                        z = pred_depth_scaled.tolist()
                        writer.write(f, z)

                except ValueError as e:
                    print("Error with: " + img_path)


