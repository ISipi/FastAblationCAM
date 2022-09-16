"""
Process:

1) Load model
2) Load model checkpoint
3) Iterate through the model and get a new model that outputs the final convolutional layer
4) Load test data and clean it up a little
5) Loop through test images and apply AblationCAM
"""

import tensorflow as tf
import os
import numpy as np
from tests.model_loading import training_model
from tests.data_import import ImportTfrecord
from src.ablation.ablation import AblationCAM

# base CNN:
network = 'densenet169'
# checkpoint location:
checkpoint_dir = os.path.join(os.path.abspath(os.curdir), r'test_model_checkpoints/densenet169/fold 5')
# number of classes:
num_class = 23

# (1) load model:
model = training_model(network, num_class, trainable=False)
ablation_instance = AblationCAM(model=model)

# (2) load model checkpoint:
ablation_instance.load_checkpoint(checkpoint_dir)

# (3) prepare the model for use:
ablation_instance.create_iterated_model()

# (4) load test data:
data_init = ImportTfrecord(network, num_class)
test_dataset = data_init.make_test_dataset()

# get the test set as a single batch:
images, labels, filenames = tf.data.experimental.get_single_element(test_dataset)

# get the class names from the filenames and clean up the filenames:
class_names, non_byte_filenames = data_init.get_class_names(filenames)

# select the output folder for the images:
out_folder = os.path.join(os.path.abspath(os.curdir), f"test_output/{network}")

# (5) loop through the test images
for enum, (img, one_hot_label, filename) in enumerate(zip(images, labels, non_byte_filenames)):
    img_label = int(np.argmax(np.array(one_hot_label)))
    img = tf.expand_dims(img, axis=0)  # have to expand test image dimension to only test one image at a time
    string_label = class_names[img_label]
    ablation_instance.setup_ablation_cam(img, img_label, string_label, 0)
    ablation_instance.run_ablation_cam()

    output = os.path.join(out_folder, f"{string_label}")
    ablation_instance.make_ablation_heatmap(output)


"""
# if counter examples are needed - loop through the test images, inside of which we loop through all possible classes
# and create heatmaps for each class.  
for enum, (img, one_hot_label, filename) in enumerate(zip(images, labels, non_byte_filenames)):
    #img_label = int(np.argmax(np.array(one_hot_label)))
    img = tf.expand_dims(img, axis=0)  # have to expand test image dimension to only test one image at a time

    for img_label in range(len(class_names)):
        string_label = class_names[img_label]
        ablation_instance.setup_ablation_cam(img, img_label, string_label, 0)
        ablation_instance.run_ablation_cam()

        output = os.path.join(out_folder, f"{string_label}")
        ablation_instance.make_ablation_heatmap(output)
        
# currently, the method does not take magnitude of the change in 
"""
