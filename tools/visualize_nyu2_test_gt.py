import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize Test GT')
    parser.add_argument('--outdir', type=str, default='vis_gt', help='Output directory')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def change_scale(depth: Image.Image, scale_size_min=240, mode=Image.NEAREST):
    w, h = depth.size
    if w < h:
        ow = scale_size_min
        oh = int(ow * h / w)
    else:
        oh = scale_size_min
        ow = int(oh * w / h)
    return depth.resize((ow, oh), mode)


def center_crop(depth: Image.Image, size=(304, 228)):
    center_crop_w, center_crop_h = size
    w1, h1 = depth.size
    x1 = int(round((w1 - center_crop_w) / 2.))
    y1 = int(round((h1 - center_crop_h) / 2.))
    return depth.crop((x1, y1, x1 + center_crop_w, y1 + center_crop_h))


def main(config: DictConfig,
         outdir: str = './visualized_gt',
         debug: bool = False):
    os.makedirs(outdir, exist_ok=False)
    test_df = pd.read_csv(config.DATASET.TEST_CSV, header=None,
                          names=['image', 'depth'])

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        depth = Image.open(row['depth'])
        depth1 = change_scale(depth,
                              scale_size_min=config.DATA.SCALE_SIZE_MIN,
                              mode=Image.NEAREST)
        depth2 = center_crop(depth1,
                             size=config.DATA.CENTER_CROP_SIZE)

        depth_array = np.array(depth2)
        plt.imshow(depth_array)
        plt.axis('off')
        plt.savefig(os.path.join(outdir, f'vis_gt_{i:05}.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        if debug:
            print('### DEBUG MODE ###')
            print(depth.size, depth1.size, depth2.size, sep='\n')
            print('exit.')
            break


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main(config, outdir=args.outdir, debug=args.debug)
