import os, csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from vnl.lib.utils.net_tools import load_ckpt
from vnl.lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
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

    with open(test_args.dataroot + "/SUNRGBD/SUNRGBD_paths.csv", "r") as csv_f:
        csv_reader = csv.DictReader(csv_f)

        for i, row in tqdm(enumerate(csv_reader)):
            with torch.no_grad():
                path = row['rgbpath']
                img = cv2.imread(os.path.join(test_args.dataroot, path))
                img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
                img_torch = scale_torch(img_resize, 255)
                img_torch = img_torch[None, :, :, :].cuda()

                _, pred_depth_softmax= model.module.depth_model(img_torch)
                pred_depth = bins_to_depth(pred_depth_softmax)
                pred_depth = pred_depth.cpu().numpy().squeeze()
                #pred_depth = (pred_depth / pred_depth.max() * 60000).astype(np.uint16)  # scale 60000 for visualization
                pred_depth_scaled = (pred_depth * 60000).astype(np.uint16)

                out_path = os.path.join(test_args.dataroot, row['rgbpath'].replace('image', 'VNL_Monocular'))
                out_path = os.path.splitext(out_path)[0] + ".png"
                out_dir = os.path.dirname(out_path)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                if np.any(pred_depth > 1.0):
                    print("Possible clipping on: " + outpath)

                cv2.imwrite(out_path, pred_depth_scaled)
                #depth = Image.fromarray(pred_depth).convert("L")
                #with open(out_path, 'wb') as fp:
                #    depth.save(fp, "JPEG")
