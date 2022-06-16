# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import glob

import pdb
import torch
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.alter_trainer import do_train as alternative_train
from maskrcnn_benchmark.engine.stage_trainer import do_train as multi_stage_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
import shutil


def removekey(d, prefix):
    r = dict(d)
    listofkeys = []
    for key in r.keys():
        if key.startswith(prefix):
            listofkeys.append(key)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


def train(cfg, local_rank, distributed, zero_shot, skip_optimizer_resume=False, save_config_path = None):

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0 #<TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )
    if cfg.TEST.DURING_TRAINING:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None
    
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    if cfg.MODEL.LINEAR_PROB:
        assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
        if hasattr(model.backbone, 'fpn'):
            assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.LINEAR_PROB:
        if model.rpn is not None:
            for key, p in model.rpn.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
        if model.roi_heads is not None:
            for key, p in model.roi_heads.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
    if cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
        if model.rpn is not None:
            for key, p in model.rpn.named_parameters():
                if 'tunable_linear' in key:
                    p.requires_grad = True

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(skip_optimizer=skip_optimizer_resume)
        arguments.update(extra_checkpoint_data)
    else:
        state_dict = checkpointer._load_file(try_to_find(cfg.MODEL.WEIGHT))
        checkpointer._load_model(state_dict)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    meters = MetricLogger(delimiter="  ")

    if zero_shot:
        return model
    
    if is_main_process():
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, " : Not Frozen")
            else:
                print(name, " : Frozen")
    report_freeze_options(cfg)
    if cfg.DATASETS.ALTERNATIVE_TRAINING:
        alternative_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    elif cfg.DATASETS.MULTISTAGE_TRAINING:
        arguments['epoch_per_stage'] = cfg.SOLVER.MULTI_MAX_EPOCH
        multi_stage_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    else:
        meters = MetricLogger(delimiter="  ")
        do_train(
            cfg,
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            data_loaders_val,
            meters=meters
        )

    return model


def test(cfg, model, distributed, verbose=False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    log_dir = cfg.OUTPUT_DIR
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
            box_only=cfg.MODEL.RPN_ONLY and cfg.MODEL.RPN_ARCHITECTURE=="RPN",
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg
        )
        synchronize()
    if verbose:
        with open(os.path.join(output_folder, "bbox.csv")) as f:
            print(f.read())

def tuning_highlevel_override(cfg,):
    if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "full":
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = False
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "linear_prob":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = True
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.DYHEAD.USE_CHECKPOINT = False # Disable checkpoint
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v1":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v2":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v3":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False # Turn on language backbone
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v4":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    return cfg

def report_freeze_options(cfg):
    print("Backbone Freeze:", cfg.MODEL.BACKBONE.FREEZE)
    print("FPN Freeze:", cfg.MODEL.FPN.FREEZE)
    print("RPN Freeze:", cfg.MODEL.RPN.FREEZE)
    print("Linear Probe:", cfg.MODEL.LINEAR_PROB)
    print("Language Freeze:", cfg.MODEL.LANGUAGE_BACKBONE.FREEZE)
    print("Linear Layer (True Prmopt Tuning):", cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER)
    print("High Level Override:", cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ft-tasks",
        default="",
        metavar="FILE",
        help="path to fine-tune configs",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-train",
        dest="skip_train",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--skip_optimizer_resume", action="store_true")

    parser.add_argument("--custom_shot_and_epoch_and_general_copy", default=None, type=str)

    parser.add_argument("--shuffle_seeds", default=None, type=str)

    parser.add_argument("--evaluate_only_best_on_test", action="store_true") # just a dummpy parameter; only used in eval_all.py, add it here so it does not complain...
    parser.add_argument("--push_both_val_and_test", action="store_true") # just a dummpy parameter; only used in eval_all.py, add it here so it does not complain...

    parser.add_argument('--use_prepared_data', action='store_true')


    parser.add_argument("--keep_testing", action="store_true")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    print(cfg)
    print("args.opts", args.opts)
    cfg.merge_from_list(args.opts)


    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    #logger.info("Collecting env info (might take some time)")
    #logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    ft_configs = []
    if args.ft_tasks:
        for ft_file in args.ft_tasks.split(","):
            for file in sorted(glob.glob(ft_file)):
                ft_configs.append(file)
    else:
        ft_configs = [args.config_file]

    shuffle_seeds = []
    if args.shuffle_seeds:
        shuffle_seeds = [int(seed) for seed in args.shuffle_seeds.split(',')]
    else:
        shuffle_seeds = [None]
    
    model = None
    for task_id, ft_cfg in enumerate(ft_configs, 1):
        for shuffle_seed in shuffle_seeds:
            cfg_ = cfg.clone()
            cfg_.defrost()
            cfg_.merge_from_file(ft_cfg)
            cfg_.merge_from_list(args.opts)
            ft_output_dir = output_dir + '/ft_task_{}'.format(task_id)

            if args.custom_shot_and_epoch_and_general_copy:
                custom_shot = int(args.custom_shot_and_epoch_and_general_copy.split("_")[0])
                custom_epoch = int(args.custom_shot_and_epoch_and_general_copy.split("_")[1])
                custom_copy = int(args.custom_shot_and_epoch_and_general_copy.split("_")[2])
                cfg_.SOLVER.MAX_EPOCH = custom_epoch
                cfg_.DATASETS.GENERAL_COPY = custom_copy
                if args.use_prepared_data:
                    if custom_shot != 0: # 0 means full data training
                        cfg_.DATASETS.TRAIN = ("{}_{}_{}".format(cfg_.DATASETS.TRAIN[0], custom_shot, cfg_.DATASETS.SHUFFLE_SEED), )
                        try:
                            custom_shot_val = int(args.custom_shot_and_epoch_and_general_copy.split("_")[3])
                        except:
                            custom_shot_val = custom_shot
                        cfg_.DATASETS.TEST = ("{}_{}_{}".format(cfg_.DATASETS.TEST[0], custom_shot_val, cfg_.DATASETS.SHUFFLE_SEED), )
                        if custom_shot_val == 1 or custom_shot_val == 3:
                            cfg_.DATASETS.GENERAL_COPY_TEST = 4 # to avoid less images than GPUs
                else:
                    cfg_.DATASETS.FEW_SHOT = custom_shot
            else:
                custom_shot = None
                custom_epoch = None

            if shuffle_seed is not None:
                cfg_.DATASETS.SHUFFLE_SEED = shuffle_seed
                ft_output_dir = ft_output_dir + '_seed_{}'.format(shuffle_seed)

            # Remerge to make sure that the command line arguments are prioritized
            cfg_.merge_from_list(args.opts)
            if "last_checkpoint" in cfg_.MODEL.WEIGHT:
                with open(cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", "last_checkpoint"), "r") as f:
                    last_checkpoint = f.read()
                cfg_.MODEL.WEIGHT = cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", last_checkpoint)
                print("cfg.MODEL.WEIGHT ", cfg_.MODEL.WEIGHT)

            mkdir(ft_output_dir)
            cfg_.OUTPUT_DIR = ft_output_dir
            
            tuning_highlevel_override(cfg_)
            cfg_.freeze()

            logger.info("Loaded fine-tune configuration file {}".format(ft_cfg))
            with open(ft_cfg, "r") as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)

            output_config_path = os.path.join(ft_output_dir, 'config.yml')
            print("Saving config into: {}".format(output_config_path))
            # save config here because the data loader will make some changes
            save_config(cfg_, output_config_path)
            logger.info("Training {}".format(ft_cfg))
            if custom_shot == 10000:
                if is_main_process():
                    print("Copying pre-training checkpoint")
                    shutil.copy(try_to_find(cfg_.MODEL.WEIGHT), os.path.join(ft_output_dir, "model_best.pth"))
            else:
                model = train(
                    cfg_, 
                    args.local_rank, 
                    args.distributed, 
                    args.skip_train or custom_shot == 10000, 
                    skip_optimizer_resume=args.skip_optimizer_resume,
                    save_config_path=output_config_path)
                
                if not args.skip_test:
                    test(cfg_, model, args.distributed)
                
                if args.keep_testing:
                    # for manual testing
                    cfg_.defrost()
                    cfg_.DATASETS.TEST = ("test", )
                    test(cfg_, model, args.distributed, verbose=True)
                    print(cfg_.DATASETS.OVERRIDE_CATEGORY)
                    pdb.set_trace()
                    # test(cfg_, model, args.distributed, verbose=True)
                    continue
                


if __name__ == "__main__":
    main()
