import argparse
import yaml
import os
import sys
import glob
import torch
import torch.nn as nn
import torchvision.utils
from contextlib import suppress
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from config import cfg, resolve_data_config, pop_unused_value
from models import create_model, resume_checkpoint, convert_splitbn_model, load_checkpoint
from optim import create_optimizer
from utils import *
from utils import ApexScaler, NativeScaler
from utils.logger import setup_test_logging
from utils.flops_counter import get_model_complexity_info
from evaler.evaler_features_multitask import Evaler

if cfg.amp == True:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

torch.backends.cudnn.benchmark = True

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Imagenet Model')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def setup_model():
    model = create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=1000,
        drop_rate=cfg.model.drop,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=cfg.model.drop_path if 'drop_path' in cfg.model else None,
        drop_block_rate=cfg.model.drop_block if 'drop_block' in cfg.model else None,
        global_pool=cfg.model.gp,
        bn_tf=cfg.BN.bn_tf,
        bn_momentum=cfg.BN.bn_momentum if 'bn_momentum' in cfg.BN else None,
        bn_eps=cfg.BN.bn_eps if 'bn_eps' in cfg.BN else None,
        checkpoint_path=cfg.model.initial_checkpoint)
    data_config = resolve_data_config(cfg, model=model)

    flops_count, params_count = get_model_complexity_info(model, data_config['input_size'], as_strings=True,
        print_per_layer_stat=False, verbose=False)
    print('Model %s created, flops_count: %s, param count: %s' % (cfg.model.name, flops_count, params_count))

    if cfg.BN.split_bn:
        assert cfg.augmentation.aug_splits > 1 or cfg.augmentation.resplit
        model = convert_splitbn_model(model, max(cfg.augmentation.aug_splits, 2))
    model.cuda()
    return model, data_config

def setup_resume(local_rank, model, optimizer):
    loss_scaler = None
    if cfg.amp == True:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
    else:
        print('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if cfg.model.resume:
        resume_epoch = resume_checkpoint(
            model, cfg.model.resume,
            optimizer=None if cfg.model.no_resume_opt else optimizer,
            loss_scaler=None if cfg.model.no_resume_opt else loss_scaler,
            log_info=local_rank == 0)

    if cfg.distributed:
        if cfg.BN.sync_bn:
            assert not cfg.BN.split_bn
            try:
                if cfg.amp:
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            except Exception as e:
                print('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if cfg.amp:
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = NativeDDP(model, device_ids=[local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    model_ema = None
    if cfg.model.model_ema == True:
        model_ema = ModelEmaV2(
            unwrap_model(model), 
            decay=cfg.model.model_ema_decay,
            device='cpu' if cfg.model.model_ema_force_cpu else None
        )
        if cfg.model.resume:
            load_checkpoint(model_ema.module, cfg.model.resume, use_ema=True)

    return model, model_ema, optimizer, resume_epoch, loss_scaler


def setup_env(args):
    if args.folder is not None:
        cfg.merge_from_file(os.path.join(args.folder, 'config_multitask_test.yaml'))
    cfg.root_dir = args.folder

    setup_test_logging()

    world_size = 1
    rank = 0  # global rank
    cfg.distributed = torch.cuda.device_count() > 1

    if cfg.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    cfg.num_gpus = world_size

    pop_unused_value(cfg)
    cfg.freeze()

    if cfg.distributed:
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (rank, cfg.num_gpus))
    else:
        print('Training with a single process on %d GPUs.' % cfg.num_gpus)
    torch.manual_seed(cfg.seed + rank)


def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    setup_env(args)

    model, data_config = setup_model()
    optimizer = create_optimizer(cfg, model)

    amp_autocast = suppress  # do nothing
    
    model, model_ema, optimizer, resume_epoch, loss_scaler = setup_resume(args.local_rank, model, optimizer)

    if cfg.distributed and cfg.BN.dist_bn in ('broadcast', 'reduce'):
        print("Distributing BatchNorm running means and vars")
        distribute_bn(model, cfg.num_gpus, cfg.BN.dist_bn == 'reduce')

    patients_id_test = []

    with open('./test_list.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            patients_id_test.append(line)

    for i in range(len(patients_id_test)):
        data_id =  patients_id_test[i]
        evaler = Evaler(data_config, data_id)

        if model_ema is not None and not cfg.model.model_ema_force_cpu:
            if cfg.distributed and cfg.BN.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, cfg.num_gpus, cfg.BN.dist_bn == 'reduce')
            evaler(model_ema.module, amp_autocast=amp_autocast)
        
        print('Generate features done!')

if __name__ == '__main__':
    main()
