


from classify_piecewise import classify




from PIL import Image
from os import listdir,path
import numpy as np

import time



def predict_folder(image_folder,top_n_images,output_prefix,
	       net_config_location,net_weights_location,
	       alpha_channel,use_gpu, win_size, crop_size):
	
	# List the names files in images_folder
	images = listdir(image_folder)
	
	# Start counting time
	start_time = time.time()

	if top_n_images is None:
		for i in range(0,len(images)):

			# Construct the image path
			image_location = path.join(image_folder,images[i])
			pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

			out = Image.fromarray(pred*255)
			out.save(output_prefix+images[i])

	else:
		for i in range(0,top_n_images):

			# Construct the image path
			image_location = path.join(image_folder,images[i])
			pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

			out = Image.fromarray(pred*255)
			out.save(output_prefix+images[i])


	#print elapsed time
	print("--- %s seconds ---" % (time.time() - start_time))









