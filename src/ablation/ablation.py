from tensorflow import keras
from typing import Any
import tensorflow as tf
import cv2
import numpy as np
from os import path, listdir, makedirs
import cmapy


class AblationLayer(keras.layers.Layer):
    def __init__(self, reference_activations: Any, ablation_value: int = 0):
        super(AblationLayer, self).__init__()
        self.filter_location = [0, 0]
        self.shape_out = reference_activations.shape
        self.reference_activations = reference_activations
        self.ablation_value = ablation_value
        self.width = 0
        self.height = 0

    def call(self, inputs) -> tf.reshape:
        """ Perform ablation. """

        self.filter_location = [self.height, self.width]
        updates = self.reference_activations.numpy()
        updates[:, self.filter_location[0], self.filter_location[1], :] *= self.ablation_value
        updates = tf.convert_to_tensor(updates)
        output = tf.reshape(updates, shape=self.shape_out)

        if self.width < self.shape_out[1]-1:
            self.width += 1
        else:
            self.width = 0
            if self.height < self.shape_out[2]-1:
                self.height += 1
            else:
                self.height = 0

        return output


class AblationCAM:
    def __init__(self, model: Any):

        # Initialize class:
        self.model = model

        # load_checkpoint init:
        self.status = None

        # create_iterated_model init:
        self.iterated_model = None
        self.select_conv_layer = None
        self.conv_layers = None

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

    def load_checkpoint(self, checkpoint_dir: str, checkpoint_name: str = None,
                        only_look_for_last_save: bool = True) -> None:
        """
        If only_look_for_last is set to True, this method will look for a checkpoint with 'best_weights' in it and
        cut at the first '.' character. Else, you must provide a checkpoint_name.

        :param checkpoint_dir: the folder where the checkpoints are located
        :param checkpoint_name: the name of the checkpoint file
        :param only_look_for_last_save: switch to False if the checkpoint file doesn't include 'best_weights' in the
        name. Otherwise use True.
        :return: None
        """

        if only_look_for_last_save:
            for i in listdir(checkpoint_dir):
                if "best_weights" in i:
                    checkpoint_name = i[:i.index(".")]
        optimizer = keras.optimizers.SGD(learning_rate=0.005)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
        manager = tf.train.CheckpointManager(checkpoint,
                                             directory=checkpoint_dir,
                                             checkpoint_name=checkpoint_name,
                                             max_to_keep=1)
        self.status = checkpoint.restore(manager.latest_checkpoint)

    def create_iterated_model(self, select_conv_layer: int = -1) -> None:
        """
        Loops through the model and finds the final layer with either 'conv' or 'convolution' in its name.

        :param select_conv_layer: -1 gives the last convolutional layer, use integers if interested in another layer
        :return: None
        """

        self.select_conv_layer = select_conv_layer
        self.conv_layers = []
        for layer in self.model.layers:
            if 'conv' in layer.name or 'convolution' in layer.name:
                self.conv_layers.append(layer.name)

        last_conv_layer = self.model.get_layer(self.conv_layers[self.select_conv_layer])
        self.iterated_model = keras.Model(self.model.inputs, [self.model.output, last_conv_layer.output])

    def setup_ablation_cam(self, input_img: Any, img_label: int, str_label: str, ablation_value: int = 0) -> None:
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
        self.reference_model_out, self.reference_activations = self.iterated_model(self.img)
        self.shape_out = self.reference_activations.shape
        self.ablation_value = ablation_value

    def run_ablation_cam(self) -> None:
        """
        Run Ablation-CAM. Collects the difference in model prediction scores between the unmodified and modified models
        for the same image at each location of the final convolutional layer's activations.

        :return: None
        """

        self.class_score_change = []
        class_out = self.reference_model_out[:, self.img_label]
        modified_model = self.ablation_cam(self.reference_activations, self.ablation_value)
        for j in range(self.shape_out[1]*self.shape_out[2]):
            modified_model_out = modified_model(self.img)
            modified_class_out = modified_model_out[:, self.img_label]
            self.class_score_change.append(float(class_out - modified_class_out))
            print(modified_class_out)

    def ablation_cam(self, reference_activations: Any, ablation_value: int) -> Any:
        """
        This method does the heavy lifting of Ablation-CAM by injecting the ablation layer after the last convolutional
        layer and ties the different parts of the model together.

        :param reference_activations: the baseline activations for a convolutional layer
        :param ablation_value: an integer value used as a mask for the activations
        :return: the modified model
        """

        # we only want to modify the original model, not replace:
        model_to_modify = self.model

        # get the output from
        find_conv_layer = model_to_modify.get_layer(self.conv_layers[self.select_conv_layer])
        new_layer = AblationLayer(reference_activations, ablation_value)(find_conv_layer.output)

        layers_after_modification = []
        for layer in reversed(model_to_modify.layers):
            if layer.name != self.conv_layers[self.select_conv_layer]:
                layers_after_modification.append(layer.name)
            else:
                break

        x = new_layer
        for layer_name in reversed(layers_after_modification):
            if 'concat' in layer_name:
                x = model_to_modify.get_layer(layer_name)([x])
            x = model_to_modify.get_layer(layer_name)(x)
        modified_model = keras.models.Model(model_to_modify.inputs, [x])
        return modified_model

    def make_ablation_heatmap(self, output_folder: str = "./ablationcam_output/") -> None:
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
