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
import utils
from skimage.measure import label,regionprops
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

class AerialDataset(utils.Dataset):

    def load_aerial(self, dataset_dir, subset, subimage_list=[(0,0)], town_list=None, 
        image_per_town=None):
        """Load a subset of the AERIAL dataset.
        dataset_dir: The root directory of the AERIAL dataset.
        subset: What to load (train, val test)
        """
        #self.subimage = subimage
        # Path
        if subset=="train":
            if town_list is None:
                town_list = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
            first_image = 1
            if image_per_town is None:
                image_per_town=36
            self.image_dir = os.path.join(dataset_dir, "train/images")
            self.mask_dir = os.path.join(dataset_dir, "train/gt")
        
        elif subset=="test":
            if town_list is None:
                town_list = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
            first_image = 1
            if image_per_town is None:
                image_per_town=36
            self.image_dir = os.path.join(dataset_dir, "test/images")
        
        elif subset=="val":
            if town_list is None:
                town_list = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
            first_image = 6
            if image_per_town is None:
                image_per_town=31
            self.image_dir = os.path.join(dataset_dir, "train/images")
            self.mask_dir = os.path.join(dataset_dir, "train/gt")
        

        # Add classes
        self.add_class("aerial", 1, "building")

        # All images or a subset?
        self.image_names = []
        for town in town_list:
            j=first_image
            while j<=first_image+(image_per_town-1):
                file = town+str(j)+".tif"
                if file in os.listdir(self.image_dir):
                    self.image_names.append(file)
                    j+=1

        # Add images
        for i, name in enumerate(self.image_names):
            for subimage in subimage_list:
                image_name_split = name.split(".")
                new_name = ".".join([image_name_split[0]+'_'+str(subimage[0])+str(subimage[1]), image_name_split[1]])
                self.add_image(
                    "aerial", image_id=i, image_name=new_name,
                    path=os.path.join(self.image_dir, name),
                    width=1024,
                    height=1024,
                    mask_path=os.path.join(self.mask_dir, name),
                    subimage=subimage)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info["path"]
        (subimx,subimy) = info['subimage']
        if subimx<4 and subimy<4:
            image_read = imageio.imread(path)[subimy*1024:(subimy+1)*1024, subimx*1024:(subimx+1)*1024]
        elif subimx==4 and subimy!=4:
            image_read = imageio.imread(path)[subimy*1024:(subimy+1)*1024, 3976:5000]
        elif subimy==4 and subimx!=4:
            image_read = imageio.imread(path)[3976:5000, subimx*1024:(subimx+1)*1024]
        elif subimy==4 and subimx==4:
            image_read = imageio.imread(path)[3976:5000, 3976:5000]
        return image_read

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        full_mask: Full original mask"""
        info = self.image_info[image_id]
        mask_path = info["mask_path"]
        (subimx,subimy) = info['subimage']
        if subimx<4 and subimy<4:
            full_mask = imageio.imread(mask_path)[subimy*1024:(subimy+1)*1024, subimx*1024:(subimx+1)*1024]
        elif subimx==4 and subimy!=4:
            full_mask = imageio.imread(mask_path)[subimy*1024:(subimy+1)*1024, 3976:5000]
        elif subimy==4 and subimx!=4:
            full_mask = imageio.imread(mask_path)[3976:5000, subimx*1024:(subimx+1)*1024]
        elif subimy==4 and subimx==4:
            full_mask = imageio.imread(mask_path)[3976:5000, 3976:5000]


        #create the mask matrix
        (instance_labels, num_instances) = label(full_mask, return_num =1, connectivity=2)

        mask = np.zeros([info['height'], info['width'], num_instances], dtype=np.uint8)
        for i in range(num_instances):
            mask[:,:,i] = (instance_labels==i+1)
 
        class_ids = np.ones(num_instances)  

        return mask, class_ids.astype(np.int32)


