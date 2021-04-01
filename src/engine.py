import os
import time
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from src.sobel import Sobel
from src.metrics import evaluate_depth_metrics, evaluate_edge_metrics


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def train(model: nn.Module,
          data_loader: DataLoader,
          optimizer: Optimizer,
          loss_config: DictConfig,
          epoch: int,
          device: str = 'cuda',
          tblogger: Optional[SummaryWriter] = None):
    """ref. https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/train.py"""

    model.train()

    # func for loss
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = Sobel().to(device)

    # init
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_depth = AverageMeter()
    losses_normal = AverageMeter()
    losses_grad = AverageMeter()
    end = time.time()
    for i, batch in enumerate(data_loader):

        # prepare
        image, depth = batch['image'], batch['depth']
        image = image.to(device)
        depth = depth.to(device)
        optimizer.zero_grad()

        # forward
        output = model(image)

        # loss: depth
        loss_depth = torch.log(torch.abs(output - depth) + loss_config.ALPHA).mean()

        # loss: grad
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + loss_config.ALPHA).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + loss_config.ALPHA).mean()

        # loss: normal
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3), requires_grad=True).to(device)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        # loss
        loss = loss_depth \
            + loss_config.LAMBDA * (loss_dx + loss_dy) \
            + loss_config.MU * loss_normal

        # update
        bs = image.size(0)
        losses.update(loss.item(), bs)
        losses_depth.update(loss_depth.item(), bs)
        losses_normal.update(loss_normal.item(), bs)
        losses_grad.update((loss_dx + loss_dy).item(), bs)

        # step
        loss.backward()
        optimizer.step()

        # time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        print(f'epoch {epoch}[{i}/{len(data_loader)}], '
              f'time {batch_time.value:.3f} ({batch_time.sum:.3f}), '
              f'loss {losses.value:.4f} ({losses.avg:.4f}), '
              f'l_d {losses_depth.value:.4f} ({losses_depth.avg:.4f}), '
              f'l_g {losses_grad.value:.4f} ({losses_grad.avg:.4f}), '
              f'l_n {losses_normal.value:.4f} ({losses_normal.avg:.4f}), ')

    if tblogger is not None:
        tblogger.add_scalar('train/loss', losses.avg, epoch + 1)
        tblogger.add_scalar('train/l_d', losses_depth.avg, epoch + 1)
        tblogger.add_scalar('train/l_g', losses_grad.avg, epoch + 1)
        tblogger.add_scalar('train/l_n', losses_normal.avg, epoch + 1)


def test(model: nn.Module,
         data_loader: DataLoader,
         threshold_edge: float = 0.25,
         device: str = 'cuda',
         epoch: Optional[int] = None,
         tblogger: Optional[SummaryWriter] = None,
         show_dir: Optional[str] = None):
    model.eval()
    get_gradient = Sobel().to(device)
    if show_dir is not None:
        os.makedirs(show_dir, exist_ok=False)

    metrics: Dict[str, AverageMeter] = {
        'MSE': AverageMeter(),
        'MAE': AverageMeter(),
        'ABS_REL': AverageMeter(),
        'LG10': AverageMeter(),
        'DELTA1': AverageMeter(),
        'DELTA2': AverageMeter(),
        'DELTA3': AverageMeter(),
        'EDGE_ACCURACY': AverageMeter(),
        'EDGE_PRECISION': AverageMeter(),
        'EDGE_RECALL': AverageMeter(),
        'EDGE_F1SCORE': AverageMeter(),
    }
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):

            # prepare
            image, depth = batch['image'], batch['depth']
            image = image.to(device)
            depth = depth.to(device)

            # forward
            output = model(image)
            output = torch.nn.functional.interpolate(output, size=[depth.size(2), depth.size(3)],
                                                     mode='bilinear', align_corners=True)

            # show output
            if show_dir is not None:
                for j, out_i in enumerate(output):
                    filename = f'vis_{i * data_loader.batch_size + j:05}.jpg'
                    plt.imshow(out_i.view(out_i.size(1), out_i.size(2)).data.cpu().numpy())
                    plt.axis('off')
                    plt.savefig(os.path.join(show_dir, filename),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

            # calc metrics
            d_metrics = evaluate_depth_metrics(output, depth)

            # forward for edge
            depth_grad_xy = get_gradient(depth)
            output_grad_xy = get_gradient(output)

            # calc edge metrics
            e_metrics = evaluate_edge_metrics(output_grad_xy, depth_grad_xy,
                                              threshold=threshold_edge)

            # update
            bs = image.size(0)
            metrics['MSE'].update(d_metrics.mse, bs)
            metrics['MAE'].update(d_metrics.mae, bs)
            metrics['ABS_REL'].update(d_metrics.abs_rel, bs)
            metrics['LG10'].update(d_metrics.lg10, bs)
            metrics['DELTA1'].update(d_metrics.delta1, bs)
            metrics['DELTA2'].update(d_metrics.delta2, bs)
            metrics['DELTA3'].update(d_metrics.delta3, bs)
            metrics['EDGE_ACCURACY'].update(e_metrics.accuracy, bs)
            metrics['EDGE_PRECISION'].update(e_metrics.precision, bs)
            metrics['EDGE_RECALL'].update(e_metrics.recall, bs)
            metrics['EDGE_F1SCORE'].update(e_metrics.f1_score, bs)

    rmse = np.sqrt(metrics['MSE'].avg)

    for k, v in metrics.items():
        print(k, v.avg, sep='\t')
    print('RMSE', rmse, sep='\t')

    if tblogger is not None:
        for k, v in metrics.items():
            tblogger.add_scalar(f'val/{k}_avg', v.avg, epoch + 1)
        tblogger.add_scalar('val/RMSE_avg', rmse, epoch + 1)
