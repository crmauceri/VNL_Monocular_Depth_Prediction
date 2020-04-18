import os
import dill
import torch
import importlib
import torch.nn as nn
from vnl.lib.core.config import cfg
from vnl.lib.utils.logging import setup_logging

logger = setup_logging(__name__)


def load_ckpt(args, model, optimizer=None, scheduler=None, val_err=[]):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        logger.info("loading checkpoint %s", args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage, pickle_module=dill)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.resume:
            args.batchsize = checkpoint['batch_size']
            args.start_step = checkpoint['step']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if 'val_err' in checkpoint:  # For backward compatibility
                val_err[0] = checkpoint['val_err']
        del checkpoint
        torch.cuda.empty_cache()


def save_ckpt(args, step, epoch, model, optimizer, scheduler, val_err={}):
    """Save checkpoint"""
    ckpt_dir = os.path.join(cfg.TRAIN.LOG_DIR, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'epoch%d_step%d.pth' %(epoch, step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'epoch': epoch,
        'batch_size': args.batchsize,
        'scheduler': scheduler.state_dict(),
        'val_err': val_err,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        save_name, pickle_module=dill)
    logger.info('save model: %s', save_name)
