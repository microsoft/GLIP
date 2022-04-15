"""Script to evaluate all checkpoints for the trained model.
OUTPUT_DIR has to contain trained checkpoints.
MODEL.WEIGHT parameter will be ignored.
"""
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_rank, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import time
import random

from tools.test_net import run_test


def main():
    parser = argparse.ArgumentParser(
        description="Script to evaluate all checkpoints in a directory"
                    "and (optionally) upload the results"
    )
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=1,
                        help="Only use checkpoints with step "
                             "divisible by eval_freq")
    parser.add_argument("--skip_eval", action="store_true",
                        help="skip evaluate ckpts")
    parser.add_argument("--overwrite_azure_logs", action="store_true",
                        dest="overwrite_azure_logs",
                        help="set this to True in order to overwrite existing"
                             "logs in azure database with results of this"
                             "experiment (matching on the --upload_with_name)")
    parser.add_argument('--collection', default='grounding',
                        help="the database collection name (grounding)")
    parser.add_argument("--evaluate_only_best_on_test", action="store_true")
    parser.add_argument("--push_both_val_and_test", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    print(args)

    if not args.skip_eval:
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)

        if args.evaluate_only_best_on_test:
            suffix_to_find = "model_best.pth"
        else:
            suffix_to_find = ".pth"
        
        if args.evaluate_only_best_on_test:
            print("Evaluating on val")
            cfg.defrost()
            cfg.DATASETS.TEST = ("val",)
            cfg.freeze()

        filenames = [fname for fname in os.listdir(cfg.OUTPUT_DIR)
                     if fname.endswith(suffix_to_find)]
        if not filenames:
            logger.info("No checkpoints found")
            return

        for fname in filenames:
            weight_path = os.path.join(cfg.OUTPUT_DIR, fname)
            log_dir = os.path.join(
                cfg.OUTPUT_DIR, "eval", os.path.splitext(fname)[0]
            )
            if log_dir:
                mkdir(log_dir)
            checkpointer = DetectronCheckpointer(
                cfg, model, save_dir=cfg.OUTPUT_DIR
            )
            if weight_path:
                _ = checkpointer.load(weight_path, force=True)
            else:
                continue
            run_test(cfg, model, distributed, log_dir)
        
        if args.evaluate_only_best_on_test:
            print("Evaluating on test")
            cfg.defrost()
            cfg.DATASETS.TEST = ("test",)
            cfg.freeze()
            for fname in filenames:
                weight_path = os.path.join(cfg.OUTPUT_DIR, fname)
                log_dir = os.path.join(
                    cfg.OUTPUT_DIR, "eval", os.path.splitext(fname)[0]
                )
                if log_dir:
                    mkdir(log_dir)
                checkpointer = DetectronCheckpointer(
                    cfg, model, save_dir=cfg.OUTPUT_DIR
                )
                if weight_path:
                    _ = checkpointer.load(weight_path, force=True)
                else:
                    continue
                run_test(cfg, model, distributed, log_dir)
            print("\n\n\nFinished testing on test\n\n\n")
    

if __name__ == '__main__':
    main()
