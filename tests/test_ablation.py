"""
Process:

1) Load test data and clean it up a little
2) Load model
3) Load model checkpoint
4) Iterate through the model and get a new model that outputs the activations of the final convolutional layer alongside
   the model scores
5) Initialize FastAblationCAM with the new, iterated model
6) Loop through test images and apply FastAblationCAM
"""

import tensorflow as tf
from os import path, curdir
import numpy as np
from tests.model_loading import training_model, PrepareModel
from tests.data_import import Polen23eTestData  # [sic]
from src.ablation.ablation import FastAblationCAM

# basic settings:
network = 'densenet169'
checkpoint_dir = path.join(path.abspath(curdir), r'test_model_checkpoints/densenet169/fold 5')
num_class = 23
out_folder = path.join(path.abspath(curdir), f"test_output/{network}")

# (1) load test data:
data_init = Polen23eTestData(network, num_class)
test_dataset = data_init.make_test_dataset()

# get the test set as a single batch and do clean up:
images, labels, filenames = tf.data.Dataset.get_single_element(test_dataset)
class_names, non_byte_filenames = data_init.get_class_names(filenames)


# (2) load model graph:
model = training_model(network, num_class, trainable=False)


# (3) load model checkpoint given the checkpoint directory:
prep_model = PrepareModel(model)
prep_model.load_checkpoint(checkpoint_dir)


# (4) prepare the model for use:
# force the model to output the activations from the last conv layer alongside class scores
prep_model.create_iterated_model(select_conv_layer=-1)

# get the new, iterated model to use as input to FastAblationCAM
model = prep_model.get_model()

# get the information regarding convolutional layers
conv_layers, last_conv_layer = prep_model.get_conv_data()


# (5) initialize FastAblationCAM
ablation_instance = FastAblationCAM(model=model, conv_layers=conv_layers, last_conv_layer=last_conv_layer)


# (6) loop through the test images
for enum, (img, one_hot_label, filename) in enumerate(zip(images, labels, non_byte_filenames)):
    img_label = int(np.argmax(np.array(one_hot_label)))

    # have to expand test image dimension when testing one image at a time
    img = tf.expand_dims(img, axis=0)
    string_label = class_names[img_label]

    ablation_instance.setup_fast_ablation_cam(img, img_label, string_label, 0)
    ablation_instance.run_fast_ablation_cam()

    output_folder = path.join(out_folder, f"{string_label}")
    ablation_instance.make_heatmap(output_folder)


"""
# if counter examples are needed - loop through the test images, inside of which we loop through all possible classes
# and create heatmaps for each class.  
for enum, (img, one_hot_label, filename) in enumerate(zip(images, labels, non_byte_filenames)):
    #img_label = int(np.argmax(np.array(one_hot_label)))
    img = tf.expand_dims(img, axis=0)  # have to expand test image dimension to only test one image at a time

    for img_label in range(len(class_names)):
        string_label = class_names[img_label]
        ablation_instance.setup_fast_ablation_cam(img, img_label, string_label, 0)
        ablation_instance.run_fast_ablation_cam()

        output = path.join(out_folder, f"{string_label}")
        ablation_instance.make_heatmap(output)
        
# currently, the method does not take magnitude of the change in class score into account when creating heatmaps


"""