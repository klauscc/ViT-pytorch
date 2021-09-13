# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)

from train import (AverageMeter, simple_accuracy, setup, set_seed)


def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                            desc="Validating... (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True,
                            disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["cifar10", "cifar100"],
                        default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type",
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir",
                        type=str,
                        default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir",
                        default="output",
                        type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every",
                        default=100,
                        type=int,
                        help="Run prediction on validation set every so many steps."
                        "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate",
                        default=3e-2,
                        type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_steps",
                        default=10000,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type",
                        choices=["cosine", "linear"],
                        default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps",
                        default=500,
                        type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level',
                        type=str,
                        default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")

    # random walk
    parser.add_argument('--rw', action='store_true', help="Whether to apply randomwalk to the last layer")
    parser.add_argument('--rw_version', type=str, default='v1', help="the randomwalk method")
    parser.add_argument('--alpha', default=0., help="alpha for random walk", type=float)
    parser.add_argument('--vis', action='store_true', help="whether to visualize the weights")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                    (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # load checkpoint
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    ckpt = torch.load(model_checkpoint)
    print(f"load checkpoint from {model_checkpoint}")
    model.load_state_dict(ckpt)

    # Prepare dataset
    _, test_loader = get_loader(args)

    # validation
    valid(args, model, test_loader)


if __name__ == "__main__":
    main()
