import tensorflow as tf
from typing import Any, List, Tuple
import os
import numpy as np
from natsort import natsorted


class ImportTfrecord:

    def __init__(self, network: str, num_classes: int):
        self.network = network
        self.num_classes = num_classes
        self.test_file = self.get_test_set()

    def get_test_set(self) -> str:
        # choose the test file
        if self.network == "inception_resnet" or self.network == "inception":
            tf_records_base_folder = f"./test_data/299_tfrecords"
        elif self.network == "nasnet_large":
            tf_records_base_folder = f"./test_data/331_tfrecords"
        else:
            tf_records_base_folder = f"./test_data/224_tfrecords"

        return os.path.join(os.path.abspath(tf_records_base_folder), 'test.tfrecords')

    @tf.function
    def preprocess(self, x: Any) -> Any:
        """ Preprocess the input images according to the pre-trained neural network's specifications """

        if self.network == 'densenet121' or self.network == 'densenet169' or self.network == 'densenet201':
            x = tf.keras.applications.densenet.preprocess_input(x)
        elif self.network == 'inception_resnet':
            x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
        elif self.network == 'inception':
            x = tf.keras.applications.inception_v3.preprocess_input(x)
        elif self.network == 'nasnet_large':
            x = tf.keras.applications.nasnet.preprocess_input(x)
        elif self.network == 'resnet50' or self.network == 'resnet101' or self.network == 'resnet152':
            x = tf.keras.applications.resnet_v2.preprocess_input(x)
        elif self.network == 'xception':
            x = tf.keras.applications.xception.preprocess_input(x)
        return x

    @tf.function
    def read_single_tfrecord(self, serialized_example: Any) -> Tuple[Any, Any, Any]:
        """ Read a serialized TFRecord with one input image.
            TFRecord must match format
            {'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'height': tf.io.FixedLenFeature([], tf.int64, default_value=224),
            'width': tf.io.FixedLenFeature([], tf.int64, default_value=224),
            'depth': tf.io.FixedLenFeature([], tf.int64, default_value=3),
            'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),}
        """

        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'height': tf.io.FixedLenFeature([], tf.int64, default_value=224),
            'width': tf.io.FixedLenFeature([], tf.int64, default_value=224),
            'depth': tf.io.FixedLenFeature([], tf.int64, default_value=3),
            'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        example = tf.io.parse_single_example(serialized_example, feature_description)
        image = tf.io.parse_tensor(example['image'], out_type=float)
        image = self.preprocess(image)
        image_shape = [example['height'], example['width'], example['depth']]
        image = tf.reshape(image, image_shape)
        label = tf.one_hot(tf.cast(example['label'], tf.uint8), depth=self.num_classes)  # one hot labels
        return image, label, example['filename']

    def make_test_dataset(self) -> tf.data.TFRecordDataset:
        """
        Returns the test dataset as a TFRecordDataset
        """

        files = tf.data.Dataset.list_files(self.test_file)
        test_dataset = tf.data.TFRecordDataset(files)
        test_dataset = test_dataset.map(map_func=lambda test_dataset: self.read_single_tfrecord(test_dataset),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # evaluate the whole test set in one batch
        c = sum(1 for _ in test_dataset)
        test_dataset = test_dataset.batch(batch_size=c, drop_remainder=False)
        print(f"Test set contains {c} images")
        return test_dataset

    @staticmethod
    def get_class_names(filenames) -> Tuple[List, List]:
        # decode filenames from bytes to UTF-8:
        bytenames = np.array(filenames)
        non_byte_filenames = [i.decode("utf-8") for i in bytenames]
        # pollen class names:
        class_names = []
        for i in non_byte_filenames:
            path = os.path.dirname(i)
            class_names.append(os.path.basename(path))
        class_names = natsorted(set(class_names))
        return class_names, non_byte_filenames
