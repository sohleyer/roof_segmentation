

from predict_folder import predict_folder


from PIL import Image
from os import listdir,path
import numpy as np


# Networks parameters
net_weights_location = "trainedModels/mlpModel.caffemodel"
net_config_location = "config/mlp.prototxt"


# Images to predict
image_folder = '/train/images/'
top_n_images = None #how many images to predict

# Output
output_prefix = '/output/'



use_gpu=0
alpha_channel = 0

win_size = 1024
crop_size = 74

predict_folder(image_folder, top_n_images, output_prefix,
               net_config_location,net_weights_location,
	       alpha_channel,use_gpu,win_size,crop_size)




