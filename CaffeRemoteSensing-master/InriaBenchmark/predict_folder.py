


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
	images = ["vienna1.tif","vienna2.tif","vienna3.tif","vienna4.tif","vienna5.tif","vienna6.tif",
             "vienna7.tif","vienna8.tif","vienna9.tif","vienna10.tif","vienna11.tif","vienna12.tif",
             "vienna13.tif","vienna14.tif","vienna15.tif","vienna16.tif","vienna17.tif","vienna18.tif",
             "vienna19.tif","vienna20.tif","vienna21.tif","vienna22.tif","vienna23.tif","vienna24.tif",
             "vienna25.tif","vienna26.tif","vienna27.tif","vienna28.tif","vienna29.tif","vienna30.tif",
             "vienna31.tif","vienna32.tif","vienna33.tif","vienna34.tif","vienna35.tif","vienna36.tif"]
	
	# Start counting time
	start_time = time.time()

	if top_n_images is None:
		for i in range(0,len(images)):

			# Construct the image path
			image_location = path.join(image_folder,images[i])
			print str(i)+ '   image location : ' + image_location
			pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

			out = Image.fromarray(pred*255)
			out.save(output_prefix+"mlp_"+images[i])

	else:
		for i in range(0,top_n_images):

			# Construct the image path
			#image_location = path.join(image_folder,images[i])
			image_location = path.join(image_folder,images[i])
			print 'image location : ' + image_location
			pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

			out = Image.fromarray(pred*255)
			#out.save(output_prefix+images[i])
			out.save(output_prefix+"mlp_"+images[i])


	#print elapsed time
	print("--- %s seconds ---" % (time.time() - start_time))









