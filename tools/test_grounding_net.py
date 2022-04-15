# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.stats import get_model_complexity_info
 
import os
import functools
import io
import os
import datetime

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    #args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 7200)
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--task_config", default=None)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )
        init_distributed_mode(args)
        print("Passed distributed init")

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0])
    if log_dir:
        mkdir(log_dir)

    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # we currently disable this
    # params, flops = get_model_complexity_info(model,
    #                                           (3, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST),
    #                                           input_constructor=lambda x: {'images': [torch.rand(x).cuda()]})
    # print("FLOPs: {}, #Parameter: {}".format(params, flops))

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

    if args.task_config:
        all_task_configs = args.task_config.split(",")
        for task_config in all_task_configs:
            cfg_ = cfg.clone()
            cfg_.defrost()
            cfg_.merge_from_file(task_config)
            cfg_.merge_from_list(args.opts)
            iou_types = ("bbox",)
            if cfg_.MODEL.MASK_ON:
                iou_types = iou_types + ("segm",)
            if cfg_.MODEL.KEYPOINT_ON:
                iou_types = iou_types + ("keypoints",)
            dataset_names = cfg_.DATASETS.TEST
            if isinstance(dataset_names[0], (list, tuple)):
                dataset_names = [dataset for group in dataset_names for dataset in group]
            output_folders = [None] * len(dataset_names)
            if log_dir:
                for idx, dataset_name in enumerate(dataset_names):
                    output_folder = os.path.join(log_dir, "inference", dataset_name)
                    mkdir(output_folder)
                    output_folders[idx] = output_folder
                data_loaders_val = make_data_loader(cfg_, is_train=False, is_distributed=distributed)

                for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                    inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=cfg_.MODEL.RPN_ONLY and (cfg_.MODEL.RPN_ARCHITECTURE == "RPN" or cfg_.DATASETS.CLASS_AGNOSTIC),
                        device=cfg_.MODEL.DEVICE,
                        expected_results=cfg_.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg_.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=output_folder,
                        cfg=cfg_
                    )
                    synchronize()
                # logger.info("FLOPs: {}, #Parameter: {}".format(params, flops))
    
    else:
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        dataset_names = cfg.DATASETS.TEST
        if isinstance(dataset_names[0], (list, tuple)):
            dataset_names = [dataset for group in dataset_names for dataset in group]
        output_folders = [None] * len(dataset_names)
        if log_dir:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(log_dir, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
            data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=cfg.MODEL.RPN_ONLY and (cfg.MODEL.RPN_ARCHITECTURE == "RPN" or cfg.DATASETS.CLASS_AGNOSTIC),
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    cfg=cfg
                )
                synchronize()
            # logger.info("FLOPs: {}, #Parameter: {}".format(params, flops))


if __name__ == '__main__':
    main()
