"""
prediction.py

This scipt allows to compute prediction for every images of the dataset
Inria Aerial Labeling Dataset. Recall that images from the MASK-RCNN (https://github.com/matterport/Mask_RCNN).
are 1024x1024. We need a particular script to generate predictions.

""" 

# Load config
# -----------------------------------------------------



# Load dataset
# -----------------------------------------------------

subimage_list=[]
for i in range(5):
    subimage_list = subimage_list + [(i,j) for j in range(5)]


# Load model
# -----------------------------------------------------

# Create model in inference mode
initial_weights = '11_mask_rcnn_aerial_0010.h5'

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

weights_path = os.path.join(AERIAL_MODEL_PATH, initial_weights)

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# -----------------------------------------------------

image_id = random.choice(dataset.image_ids)
image_id = 8
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# -----------------------------------------------------

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
masked_image_maskrcnn = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)