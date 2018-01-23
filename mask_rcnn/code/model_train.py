"""
Mask R-CNN - Training model
The main Mask R-CNN model training implemenetation.

Adapt from Waleed Abdulla by Sébastien Ohleyer
"""

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

ROOT_DIR = os.getcwd()


########################################
import aerial
config = aerial.AerialConfig()

# Local
#AERIAL_DIR = "/Users/sebastienohleyer/Documents/ENS MVA/Object recognition/AerialImageDataset/"  # TODO: enter value here
#COCO_MODEL_PATH = "~/Document/ENS MVA/Object recognition/Mask_RCNN-coco/coco_weigths/mask_rcnn_coco.h5"
#MODEL_DIR = "../trained_model/"

# Floydhub
#AERIAL_DIR = "/"  # TODO: enter value here
#COCO_MODEL_PATH = "/coco_weights/mask_rcnn_aerial_0039.h5"
#MODEL_DIR = "/output/trained_model/"
# run : floyd run --data sohleyer/datasets/aerialimagedataset_train/1:/train --data sohleyer/datasets/coco_weights/1:/coco_weights --env tensorflow-1.3 --mode jupyter

# AWS
AERIAL_DIR = "/home/ubuntu/aerialimagedataset"  # TODO: enter value here
COCO_MODEL_PATH = "/home/ubuntu/mask_rcnn/trained_model/mask_rcnn_aerial_0039.h5"
MODEL_DIR = "/home/ubuntu/mask_rcnn/output/"


TOWN_LIST = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
IMAGE_PER_TOWN = None
SUBIMAGE_LIST = [(2,0),(3,0)]


config.display()


######################################
# Load dataset train
dataset_train = aerial.AerialDataset()
dataset_train.load_aerial(dataset_dir=AERIAL_DIR, subset="train", subimage_list=SUBIMAGE_LIST, town_list=TOWN_LIST, image_per_town=IMAGE_PER_TOWN)
dataset_train.prepare()

print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Info: {}".format(dataset_train.class_info))


#####################################
# Load dataset val
dataset_val = aerial.AerialDataset()
dataset_val.load_aerial(dataset_dir=AERIAL_DIR, subset="val", subimage_list=SUBIMAGE_LIST, town_list=TOWN_LIST, image_per_town=IMAGE_PER_TOWN)
dataset_val.prepare()

print("Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Info: {}".format(dataset_val.class_info))


#####################################
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True) # WHICH IS NOT COCO ANYMORE


#####################################
# Stage 1 : Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')

# Stage 2
# Finetune layers from ResNet stage 4 and up
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='4+')

# Stage 3 : Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=40, layers="all")


