import argparse
import os
from typing import Optional

import torch
from omegaconf import DictConfig
from tensorboardX import SummaryWriter

from src.data import build_data_loader
from src.engine import train, test
from src.utils import load_config, print_config, make_deterministic, prepare_training_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a predictor')
    parser.add_argument('--config', type=str, default=None, help='train config file path')
    parser.add_argument('--out-dir', type=str, default=None, help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overwrite configs. (ex. SOLVER.NUM_WORKERS=8)')
    return parser.parse_args()


def main(config: DictConfig,
         out_dir: str = 'results',
         resume_from: Optional[str] = None):

    # seed
    if config.SEED is not None:
        make_deterministic(seed=config.SEED)

    # data
    train_loader, test_loader = build_data_loader(config)

    # training modules
    start_epoch, model, optimizer, scheduler = \
        prepare_training_modules(config, resume_from=resume_from)

    # output setting
    os.makedirs(out_dir, exist_ok=True)
    tblogger = SummaryWriter(out_dir)

    # train
    for epoch in range(start_epoch, config.SOLVER.EPOCH):
        lr = scheduler.get_last_lr()[0]
        print(f'#### Epoch{epoch}, lr: {lr} ####')
        tblogger.add_scalar('train/lr', lr, epoch)

        train(model=model,
              data_loader=train_loader,
              optimizer=optimizer,
              loss_config=config.LOSS,
              epoch=epoch,
              device=config.DEVICE,
              tblogger=tblogger)
        test(model=model,
             data_loader=test_loader,
             epoch=epoch,
             device=config.DEVICE,
             tblogger=tblogger,
             threshold_edge=config.TEST.THRESHOLD_EDGE)
        scheduler.step()

        # save
        ckpt = {'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}
        torch.save(ckpt, os.path.join(out_dir, f"snapshot.ckpt"))
        if (epoch + 1) % config.SOLVER.SAVE_INTERVAL == 0:
            torch.save(ckpt, os.path.join(out_dir, f"epoch_{epoch + 1}.ckpt"))


if __name__ == "__main__":
    args = parse_args()

    # load config
    config = load_config(args.config, update_dotlist=args.opts)
    print_config(config)

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config.OUTPUT_DIR

    main(config, out_dir, args.resume)
