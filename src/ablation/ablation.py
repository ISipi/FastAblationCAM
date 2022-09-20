from tensorflow import keras
from typing import Any, List
import tensorflow as tf
import cv2
import numpy as np
from os import path, listdir, makedirs
import cmapy


class FastAblationLayer(keras.layers.Layer):
    def __init__(self, reference_activations: Any, ablation_value: int = 0, trainable: bool = False, *args, **kwargs):
        super(FastAblationLayer, self).__init__()
        self.filter_location = [0, 0]
        self.output_dim = reference_activations.shape
        self.reference_activations = reference_activations
        self.ablation_value = ablation_value
        self.width = 0
        self.height = 0
        self.trainable = trainable

    def build(self, input_shape):
        """ builds the layer by initializing the weights using the activations from the input convolutional layer """

        init = tf.constant_initializer(self.reference_activations.numpy())
        self.w = self.add_weight(name='Ablation',
                                 shape=self.output_dim, initializer=init,
                                 trainable=self.trainable)

        super(FastAblationLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        """
            updates the weights by choosing a single (width, height) location and sets all values to zero, depthwise,
            at that location.
        """

        self.filter_location = [self.height, self.width]
        updates = self.reference_activations.numpy()
        updates[:, self.filter_location[0], self.filter_location[1], :] *= self.ablation_value
        updates = tf.convert_to_tensor(updates)
        self.w = tf.reshape(updates, shape=self.output_dim)
        if self.width < self.output_dim[1]-1:
            self.width += 1
        else:
            self.width = 0
            if self.height < self.output_dim[2]-1:
                self.height += 1
            else:
                self.height = 0

        return self.w


class FastAblationCAM:
    def __init__(self, model: Any, conv_layers: List, last_conv_layer: Any):

        # Initialize class:
        self.model = model
        self.select_conv_layer = last_conv_layer
        self.conv_layers = conv_layers

        # setup_ablation_cam init:
        self.img = None
        self.img_label = None
        self.str_label = None
        self.reference_model_out = None
        self.reference_activations = None
        self.shape_out = None
        self.ablation_value = None

        # run_ablation_cam init:
        self.class_score_change = None

    def setup_fast_ablation_cam(self, input_img: Any, img_label: int, str_label: str, ablation_value: int = 0) -> None:
        """
        Give the test image and its class label (as an integer) to setup Ablation-CAM.

        :param input_img: the input image as a tensor
        :param img_label: the image label as an integer
        :param str_label: the image label as a string
        :param ablation_value: integer value by which to ablate the convolutional layer. Default 0, but positive values
                               above 1 should result in counterfactual heatmaps. Negative values should not have an
                               impact when ReLU is used in forward activations.
        :return: None
        """

        self.img = input_img
        self.img_label = img_label
        self.str_label = str_label

        # get the baseline reference class score and activations:
        self.reference_model_out, self.reference_activations = self.model(self.img)
        self.shape_out = self.reference_activations.shape
        self.ablation_value = ablation_value

    def run_fast_ablation_cam(self) -> None:
        """
        Run Ablation-CAM. Collects the difference in model prediction scores between the unmodified and modified models
        for the same image at each location of the final convolutional layer's activations.

        :return: None
        """

        self.class_score_change = []
        class_out = self.reference_model_out[:, self.img_label]
        modified_model = self.inject_fast_ablation(self.reference_activations, self.ablation_value)
        for j in range(self.shape_out[1]*self.shape_out[2]):
            modified_model_out = modified_model(self.img)
            modified_class_out = modified_model_out[:, self.img_label]
            self.class_score_change.append(float(class_out - modified_class_out))
            print(modified_class_out, f"{((j+1)/(self.shape_out[1]*self.shape_out[2]))*100:.2f} % complete")

    def inject_fast_ablation(self, reference_activations: Any, ablation_value: int) -> Any:
        """
        This method does the heavy lifting of Ablation-CAM by injecting the ablation layer after the last convolutional
        layer and ties the different parts of the model together.

        :param reference_activations: the baseline activations for a convolutional layer
        :param ablation_value: an integer value used as a mask for the activations
        :return: the modified model
        """

        # we only want to modify the original model, not replace:
        model_to_modify = self.model

        # take the activations from the chosen conv layer and initialize the FastAblation layer
        find_conv_layer = model_to_modify.get_layer(self.conv_layers[self.select_conv_layer])
        new_layer = FastAblationLayer(reference_activations, ablation_value)(find_conv_layer.output)

        # find the layers that come after the injected layer
        layers_after_modification = []
        for layer in reversed(model_to_modify.layers):
            if layer.name != self.conv_layers[self.select_conv_layer]:
                layers_after_modification.append(layer.name)
            else:
                break

        # get the model configuration - needed for finding incoming layers for concatenation layers
        model_conf = model_to_modify.get_config()['layers']
        x = new_layer

        # loop through the layers after the injected layer in reverse and check whether the layer is a concatenation
        # layer. If it is a concat layer, also find its inbound layers and, if applicable, replace the selected
        # convolutional layer with the injected layer. If it's not a concat layer, just simply add after the previous.
        for layer_name in reversed(layers_after_modification):
            if 'concat' in layer_name:
                previous_layers = []
                for layer_conf in model_conf:
                    if layer_conf['name'] == layer_name:
                        for inbound_nodes in layer_conf['inbound_nodes'][0]:
                            if inbound_nodes[0] != self.conv_layers[self.select_conv_layer]:
                                previous_layers.append(model_to_modify.get_layer(inbound_nodes[0]).output)
                            else:
                                previous_layers.append(x)
                x = model_to_modify.get_layer(layer_name)(previous_layers)
            else:
                x = model_to_modify.get_layer(layer_name)(x)

        modified_model = keras.models.Model(model_to_modify.inputs, [x])
        return modified_model

    def make_heatmap(self, output_folder: str = "./ablationcam_output/") -> None:
        """
        Create the heatmaps.

        :param output_folder: the output folder
        :return: None
        """

        # map the change in class scores after ablation to the shape of the convolutional activations:
        score_map = np.reshape(self.class_score_change, (self.shape_out[1], self.shape_out[2]))

        # normalize the scores in the mapped values to [0, 1]
        norm_heatmap = np.maximum(score_map, 0)
        norm_heatmap /= np.max(norm_heatmap)

        # upsample the heatmap and scale the values to [0, 255]
        upsampled_heatmap = cv2.resize(norm_heatmap, (self.img.shape[1], self.img.shape[2]))
        upsampled_heatmap = np.uint8(255 * upsampled_heatmap)

        # apply colour map
        coloured_heatmap = cv2.applyColorMap(upsampled_heatmap, cmapy.cmap('bwr'))

        # prepare the original input image
        orig_img = np.array(self.img).reshape((self.img.shape[1], self.img.shape[2], 3))
        norm_image = cv2.normalize(orig_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)

        # overlay the heatmap over the original input image
        gray = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        base_img = np.zeros_like(norm_image)
        for i in range(3):
            base_img[:, :, i] = gray
        overlay_intensity = 0.5
        combined_img = cv2.addWeighted(coloured_heatmap, overlay_intensity, base_img, overlay_intensity, 0)

        # save the image in output_folder
        if not path.exists(output_folder):
            makedirs(output_folder)
        count_of_files = len([i for i in listdir(output_folder) if 'overlay' in i])
        new_img_label = f'overlay_{count_of_files + 1}_{self.str_label}.png'
        overlay_name = path.join(output_folder, new_img_label)
        cv2.imwrite(overlay_name, combined_img)
