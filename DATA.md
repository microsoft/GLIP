We provide guidance for preparing the data used by GLIP. Note that not all data are needed for a specific experiments. Please check the `` Required Data`` fields in [README](README.md) to download necessary data. All data should by placed under the ``DATASET`` folder.


#### ``COCO``
Download the original [COCO](https://cocodataset.org/#download) data into ``DATASET/coco`` folder. The contents should be organized as follows:

###### train2017
    DATASET/coco/train2017
    DATASET/coco/annotations/instances_train2017.json

###### val2017
    DATASET/coco/val2017
    DATASET/coco/annotations/instances_val2017.json
###### test2017
    DATASET/coco/test2017
    DATASET/coco/annotations/image_info_test-dev2017.json
###### train2014
    DATASET/coco/train2014

#### ``LVIS``
LVIS use the same images as COCO. Thus prepare the COCO images first.

    DATASET/coco

Download the following annotation files:

    "wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_v1_minival_inserted_image_name.json -O DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json"
    "wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_od_val.json -O coco/annotations/lvis_od_val.json"

#### ``Object Detection in the Wild (ODinW)``
Please see the "ODinW / Custom Dataset Evaluation" section in [README.md](README.md) for preparing the Aquarium dataset. We will release all the data in ODinW in the next update.


#### ``Objects365``
We store Objects365 data in the TSV format. Please see [link](https://github.com/microsoft/scene_graph_benchmark/tree/main/tools/mini_tsv) for a description of the TSV format. We provide the annotation files:

    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/objects365_train_vgoiv6.cas2000.yaml -O DATASET/Objects365/objects365_train_vgoiv6.cas2000.yaml
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/train.label.tsv -O DATASET/Objects365/train.label.tsv
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/train.label.linelist.cas.2000.tsv -O DATASET/Objects365/train.label.linelist.cas.2000.tsv
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/train.label.lineidx -O DATASET/Objects365/train.label.lineidx
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/train.hw.tsv -O DATASET/Objects365/train.hw.tsv
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/train.hw.lineidx -O DATASET/Objects365/train.hw.lineidx
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/Objects365/object365_vgoiv6_class2ind.json -O DATASET/Objects365/object365_vgoiv6_class2ind.json

We cannot host the image data. Please download the original image data and organize them into ``DATASET/Objects365/images.tsv`` and ``DATASET/Objects365/images.lineidx``.
    
#### ``Flickr30K``
Download the Flickr30K images from [Link](http://shannon.cs.illinois.edu/DenotationGraph/) and put them under ``DATASET/flickr30k/flickr30k_images/``. Download the [MDETR annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and put them under ``DATASET/mdetr_annotations/``. The dataset structure should look like:

    DATASET/flickr30k/flickr30k_images/
    DATASET/mdetr_annotations/final_flickr_separateGT_*

#### ``MixedGrounding``
This is the grounding dataset curated by [MDETR](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md). 
Please prepare the COCO train2014 data and put them under ``DATASET/coco/train2014``. 
Prepare the [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) and put them under ``DATASET/gqa/images/``. 

Then download the annotation files. The original MDETR annotation file contains COCO images; we provide a version without COCO images: ``wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/mdetr_annotations/final_mixed_train_no_coco.json -O DATASET/mdetr_annotations/final_mixed_train_no_coco.json``.

The dataset structure should look like:

    "DATASET/coco/train2014" 
    "DATASET/gqa/images"
    "DATASET/mdetr_annotations/final_mixed_train_no_coco.json",

#### ``GCC``
Goolge conceptual captions with pseudo-grounding annotations.
To be released in the next update.