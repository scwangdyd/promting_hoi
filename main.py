import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--clip_model', default="ViT-B/16", type=str,
                        help="Name of pretrained CLIP model")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Vision
    parser.add_argument('--embed_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--image_resolution', default=224, type=int,
                        help="input image resolution to the vision transformer")
    parser.add_argument('--vision_layers', default=12, type=int,
                        help="number of layers in vision transformer")
    parser.add_argument('--vision_width', default=768, type=int,
                        help="feature channels in vision transformer")
    parser.add_argument('--vision_patch_size', default=16, type=int,
                        help="patch size: the input image is divided into multiple patches")
    parser.add_argument('--hoi_token_length', default=5, type=int,
                        help="Number of learnable hoi tokens added to transformer's input")
    # * Text
    parser.add_argument('--context_length', default=77, type=int,
                        help="Maximum length of the text description")
    parser.add_argument('--vocab_size', default=49408, type=int,
                        help="")
    parser.add_argument('--transformer_width', default=512, type=int,
                        help="")
    parser.add_argument('--transformer_heads', default=8, type=int,
                        help="")
    parser.add_argument('--transformer_layers', default=12, type=int,
                        help="")
    parser.add_argument('--prefix_length', default=8, type=int,
                        help="Length the of the learnable prefix in the sentence")
    parser.add_argument('--conjun_length', default=2, type=int,
                        help="Length of the conjunction words between actions and objects")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=36, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Frozen CLIP model
    update_modules = set([n for n, p in model_without_ddp.named_parameters() if "hoi" in n or "bbox" in n])
    frozen_modules = set([n for n, p in model_without_ddp.named_parameters() if n not in update_modules])
    
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if n in update_modules and p.requires_grad],
            "lr": args.lr
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if n in frozen_modules and p.requires_grad],
            "lr": 0.,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print(f"load checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # checkpoint['optimizer']["param_groups"][0]["lr"] = 0.0001
            optimizer.load_state_dict(checkpoint['optimizer'])
            # checkpoint["lr_scheduler"]["step_size"] = args.lr_drop
            # checkpoint["lr_scheduler"]["_last_lr"] = [0.0001, 0.0]
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, evaluator = evaluate(model, criterion, data_loader_val, device, args)
        # if args.output_dir:
        #     utils.save_on_master(evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)