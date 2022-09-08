from tensorflow import keras
from typing import Any, List
from tests.model_loading import LoadModels
import tensorflow as tf
import cv2
import numpy as np
from os import path, listdir, makedirs
import cmapy


class AblationLayer(keras.layers.Layer):
    def __init__(self, shape_out: Any, filter_location: Any, reference_activations: Any, ablation_value: int = 0):
        super(AblationLayer, self).__init__()
        self.filter_location = filter_location
        self.shape_out = shape_out
        self.reference_activations = reference_activations

        self.ablation_value = ablation_value

    def call(self, inputs) -> Any:
        """ Perform ablation """
        updates = self.reference_activations.numpy()
        updates[:, self.filter_location[0], self.filter_location[1], :] *= self.ablation_value
        updates = tf.convert_to_tensor(updates)
        output = tf.reshape(updates, shape=self.shape_out)
        return output


class AblationCAM(LoadModels):
    def __init__(self, *args, **kwargs):
        super(AblationCAM, self).__init__(*args, **kwargs)
        self.load_checkpoint()

    def setup_ablation_cam(self, input_img: Any, img_label: int, ablation_value: int = 0) -> None:
        """
        Give the test image and its class label (as an integer) to setup Ablation-CAM
        :param input_img:
        :param img_label:
        :param ablation_value: integer value by which to ablate the convolutional layer. Default 0, but positive values
                               above 1 should result in counterfactual heatmaps. Negative values should not have an
                               impact when ReLU is used in forward activations.
        :return:
        """
        self.img = input_img
        self.img_label = img_label
        iterated_model = self.get_iterated_model(-1)
        self.reference_model_out, self.reference_activations = iterated_model(self.img)
        self.shape_out = self.reference_activations.shape
        self.ablation_value = ablation_value

    def run_ablation_cam(self) -> None:
        """
        Run Ablation-CAM.
        :return:
        """
        class_out = self.reference_model_out[:, self.img_label]
        self.class_score_change = []
        for j in range(self.shape_out[1]):
            for i in range(self.shape_out[2]):
                iterated_model = self.ablation_cam(self.shape_out, [j, i], self.reference_activations,
                                                   self.ablation_value)
                modified_model_out, modified_activations = iterated_model(self.img)
                modified_class_out = modified_model_out[:, self.img_label]
                self.class_score_change.append(float(class_out - modified_class_out))
                print(modified_class_out)

    def make_ablation_heatmap(self, output_folder: str = "./ablationcam_output/") -> None:
        """
        Give an output folder where the heatmaps overlaid on top of the input images are placed.
        :param output_folder:
        :return:
        """
        score_map = np.reshape(self.class_score_change, (self.shape_out[1], self.shape_out[2]))

        norm_heatmap = np.maximum(score_map, 0)
        norm_heatmap /= np.max(norm_heatmap)

        upsampled_heatmap = cv2.resize(norm_heatmap, (self.img.shape[1], self.img.shape[1]))
        upsampled_heatmap = np.uint8(255 * upsampled_heatmap)

        coloured_heatmap = cv2.applyColorMap(upsampled_heatmap, cmapy.cmap('bwr'))

        orig_img = np.array(self.img).reshape((self.img.shape[1], self.img.shape[1], 3))
        norm_image = cv2.normalize(orig_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)

        gray = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        base_img = np.zeros_like(norm_image)
        base_img[:, :, 0] = gray
        base_img[:, :, 1] = gray
        base_img[:, :, 2] = gray
        overlay_intensity = 0.5
        combined_img = cv2.addWeighted(coloured_heatmap, overlay_intensity, base_img, overlay_intensity, 0)

        output_folder = output_folder
        if not path.exists(output_folder):
            makedirs(output_folder)
        count_of_files = len([i for i in listdir(output_folder) if 'overlay' in i])
        new_img_label = f'overlay_{count_of_files + 1}.png'
        overlay_name = path.join(output_folder, new_img_label)
        cv2.imwrite(overlay_name, combined_img)

    def ablation_cam(self, shape_out, filter_location, reference_activations, ablation_value) -> Any:
        """
        This method does the heavy lifting of Ablation-CAM by injecting the ablation layer after the last convolutional
        layer and ties the different parts of the model together.
        """

        self.last_conv_layer = self.model.get_layer(self.conv_layers[self.select_conv_layer])
        self.new_layer = AblationLayer(shape_out, filter_location, reference_activations, ablation_value)(
            self.last_conv_layer.output)  #

        layers_after_modification = []
        for layer in reversed(self.model.layers):
            if layer.name != self.conv_layers[self.select_conv_layer]:
                layers_after_modification.append(layer.name)
            else:
                break

        x = self.new_layer
        for layer_name in reversed(layers_after_modification):
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.models.Model(self.model.inputs, [x, self.new_layer])
        return classifier_model
