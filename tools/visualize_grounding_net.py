# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import yaml
import json
import pdb
import os
import random
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
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os
import functools
import io
import os
import datetime

import torch
import torch.distributed as dist

def load(url_or_file_name):
    try:
        response = requests.get(url_or_file_name)
    except:
        response = None
    if response is None:
        pil_image = Image.open(url_or_file_name).convert("RGB")
    else:
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

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

def imshow(img, file_name = "tmp.jpg"):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, "test", wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(file_name)

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

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--box_pixel', default=3,  type=int)
    parser.add_argument('--text_size', default=1, type=float)
    parser.add_argument('--text_pixel', default=1, type=int)
    parser.add_argument('--image_index', default=0, type=int)
    parser.add_argument('--threshold', default=0.6, type=float)
    parser.add_argument("--text_offset", default=10, type=int)
    parser.add_argument("--text_offset_original", default=4, type=int)
    parser.add_argument("--color", default=255, type=int)

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
    try:
        model.to(cfg.MODEL.DEVICE)
    except:
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()


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

                image_index = args.image_index
                            
                visualizer = GLIPDemo(
                    cfg,
                    min_image_size=800,
                    confidence_threshold=0.7,
                    show_mask_heatmaps=False,
                    load_model=False
                )
                output_folder = output_folders[0]
                dataset_name = dataset_names[0]
                data_loader_val = data_loaders_val[0]

                threshold = args.threshold
                image_index = args.image_index
                alpha = args.alpha
                box_pixel = args.box_pixel
                text_size = args.text_size
                text_pixel = args.text_pixel
                text_offset = args.text_offset
                text_offset_original = args.text_offset_original
                color = args.color

                
                predictions = inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=cfg_.MODEL.RPN_ONLY and (cfg_.MODEL.RPN_ARCHITECTURE == "RPN" or cfg_.DATASETS.CLASS_AGNOSTIC),
                    device=cfg_.MODEL.DEVICE,
                    expected_results=cfg_.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg_.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    cfg=cfg_,
                    visualizer=visualizer
                )
                dataset = data_loader_val.dataset
                image_id = dataset.ids[image_index]
                try:
                    image_path = os.path.join(dataset.root, dataset.coco.loadImgs(image_id)[0]["file_name"])
                    categories = dataset.coco.dataset["categories"]
                except:
                    lvis = dataset.lvis
                    img_id = dataset.ids[image_index]
                    ann_ids = lvis.get_ann_ids(img_ids=img_id)
                    target = lvis.load_anns(ann_ids)

                    image_path = "DATASET/coco/" +  "/".join(dataset.lvis.load_imgs(img_id)[0]["coco_url"].split("/")[-2:])
                    categories = dataset.lvis.dataset["categories"]

                image = load(image_path)
                no_background = True
                label_list = []
                for index, i in enumerate(categories):
                    # assert(index + 1 == i["id"])
                    if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                        label_list.append(i["name"])
                visualizer.entities =  label_list
                
                result, _ = visualizer.visualize_with_predictions(
                    image,
                    predictions, 
                    threshold,
                    alpha=alpha,
                    box_pixel=box_pixel,
                    text_size=text_size,
                    text_pixel=text_pixel,
                    text_offset=text_offset,
                    text_offset_original=text_offset_original,
                    color=color,
                )
                imshow(result, "./visualize/tmp{}.jpg".format(image_index))
                image_index += 1
                #pdb.set_trace()
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

if __name__ == '__main__':
    main()
