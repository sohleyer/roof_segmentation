"""
Mask R-CNN
Configurations and data loading code for INRIA AERIAL.

Written by Sébastien Ohleyer

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
import imageio

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils

from config import Config
# import utils
# import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class AerialConfig(Config):
    """Configuration for training on INRIA AERIAL.
    Derives from the base Config class and overrides values specific
    to the INRIA AERIAL dataset.
    """
    # Give the configuration a recognizable name
    NAME = "aerial"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class AerialDataset():
    def load_aerial(self, dataset_dir, subset, return_aerial=False):
        """Load a subset of the AERIAL dataset.
        dataset_dir: The root directory of the AERIAL dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        # Path
        self.image_dir = os.path.join(dataset_dir, "train/images" if subset == "train"
                                 else "val/images")
        self.mask_dir = os.path.join(dataset_dir, "train/gt" if subset == "train"
                                 else "val/gt")

        self.class_ids=['building', 'not building']
        self.num_classes=len(self.class_ids)

        self.image_ids = []
        for file in os.listdir(self.image_dir):
            if file.endswith(".tif"):   
                self.image_ids.append(file)

        if return_aerial:
            return self

    def load_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image_read = imageio.imread(image_path)
        return image_read

    def load_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, image_id)
        mask_read = imageio.imread(mask_path)
        return mask_read
