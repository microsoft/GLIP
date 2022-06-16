import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import yaml
import json
import pdb
import os
import random

odinw_config_root = "configs/odinw/"
all_odinw_configs = [
    'AerialMaritimeDrone_large.yaml', 'Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml', 'CottontailRabbits.yaml', 'EgoHands_generic.yaml', 'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml', 'Packages_Raw.yaml', 'Raccoon_Raccoon.v2-raw.coco.yaml', 'ShellfishOpenImages_raw.yaml', 'VehiclesOpenImages_416x416.yaml', 'pistols_export.yaml', 'pothole.yaml', 'thermalDogsAndPeople.yaml', 'Pascal2012.yaml',]


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

def imshow(img, caption, file_name = "tmp.jpg"):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(file_name)

# Use this command for evaluate the GLPT-T model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

# random choose one image from the val
shuffle = True
for config in all_odinw_configs:
    with open(os.path.join(odinw_config_root, config), 'r') as fp:
        file = yaml.load(fp, Loader=yaml.CLoader)
    val_annotation_file = "DATASET/" + file["DATASETS"]["REGISTER"]["val"]["ann_file"]
    with open(val_annotation_file, 'r') as fp:
        val_annotation = json.load(fp)
        sampled_image = random.choice(val_annotation["images"])
        print(sampled_image)
        image_path = os.path.join("DATASET", file["DATASETS"]["REGISTER"]["val"]["img_dir"], sampled_image["file_name"])

        image = load(image_path)

        if "CAPTION_PROMPT" in file["DATASETS"]:
            pass

        # generate the caption
        caption = 'bobble heads on top of the shelf'
        result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
        imshow(result, caption, "tmp.jpg")

