import tensorflow as tf
import os
import numpy as np
from tests.model_loading import training_model
from tests.data_import import ImportTfrecord
from src.ablation.ablation import AblationCAM

"""
Stages:

1) Load model
2) Load model checkpoint
3) Iterate through the model and get a new model that outputs the final convolutional layer 
4) Load test data
5) Loop through test images and apply AblationCAM
"""

# base CNN:
network = 'densenet169'
# checkpoint location:
checkpoint_dir = os.path.join(os.path.abspath(os.curdir), r'test_model_checkpoints/densenet169/fold 5')
# number of classes:
num_class = 23

# (1) load model:
model = training_model(network, num_class, trainable=False)
ablation_instance = AblationCAM(model, network, num_class, False, checkpoint_dir, checkpoint_name=None,
                                only_look_for_last=True)



# (2) load model checkpoint:
ablation_instance.load_checkpoint()

# (3) get the iterated model that also outputs the final convolutional layer:
#iterated_model, all_conv_layers, selected_conv_layer = model.get_iterated_model()

# (4) load test data:
data_init = ImportTfrecord(network, num_class)
test_dataset = data_init.make_test_dataset()

# get the test set as a single batch:
images, labels, filenames = tf.data.experimental.get_single_element(test_dataset)

# get the class names from the filenames and clean up the filenames:
class_names, non_byte_filenames = data_init.get_class_names(filenames)

# select the output folder for the images:
out_folder = os.path.join(os.path.abspath(os.curdir), f"test_output/{network}")

# create a list of dictionaries that capture the image information and the images
#lst_of_figs = []



# (5) loop through the test images
for enum, (img, one_hot_label, filename) in enumerate(zip(images, labels, non_byte_filenames)):
    img_label = int(np.argmax(np.array(one_hot_label)))
    string_label = class_names[img_label]

    img = tf.expand_dims(img, axis=0)  # have to expand test image dimension to only test one image at a time
    ablation_instance.setup_ablation_cam(img, img_label, 0)
    ablation_instance.run_ablation_cam()
    ablation_instance.make_ablation_heatmap(os.path.join(os.path.abspath(os.curdir), 'test_output'))
    """    combined_fig = dict()
        combined_fig['final_label'] = string_label
        combined_fig['input_img'] = img
        combined_fig['filename'] = filename
        combined_fig['heatmap'] = heatmap
        combined_fig['original_img'] = orig_img
        combined_fig['combined_img'] = combined_img
        lst_of_figs.append(combined_fig)
    """
#lst_of_figs = sorted(lst_of_figs, key=lambda x: x['final_label'])

