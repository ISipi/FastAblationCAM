from os import listdir

from tensorflow import train
from tensorflow import keras
from typing import List, Tuple


class LoadModels:
    def __init__(self, model, network, num_classes, trainable, checkpoint_dir, checkpoint_name: str = None,
                 only_look_for_last: bool = True):

        self.network = network
        self.class_count = num_classes
        self.trainable = trainable
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.only_look_for_last = only_look_for_last

        self.model = model
        self.last_conv_layer = None
        self.status = None
        self.select_conv_layer = None
        self.conv_layers = []

    def load_checkpoint(self) -> None:
        """
        If only_look_for_last is set to True, this method will look for a checkpoint with 'best_weights' in it and
        cut at the first '.' character. Else, you must provide a checkpoint_name.
        """
        if self.only_look_for_last:
            for i in listdir(self.checkpoint_dir):
                if "best_weights" in i:
                    self.checkpoint_name = i[:i.index(".")]
        optimizer = keras.optimizers.SGD(learning_rate=0.005)
        checkpoint = train.Checkpoint(optimizer=optimizer, model=self.model)
        manager = train.CheckpointManager(checkpoint,
                                          directory=self.checkpoint_dir,
                                          checkpoint_name=self.checkpoint_name,
                                          max_to_keep=1)
        self.status = checkpoint.restore(manager.latest_checkpoint)

    def get_iterated_model(self, select_conv_layer: int = -1) -> Tuple[keras.Model, List, int]:
        """
        Loops through the model and finds the final layer with either 'conv' or 'convolution' in its name.
        :param select_conv_layer: -1 gives the last convolutional layer, use integers if interested in another layer
        Returns an iterated model that includes the selected convolutional layer
        """
        self.select_conv_layer = select_conv_layer
        for layer in self.model.layers:
            if 'conv' in layer.name or 'convolution' in layer.name:
                self.conv_layers.append(layer.name)
        self.last_conv_layer = self.model.get_layer(self.conv_layers[self.select_conv_layer])
        iterated_model = keras.Model(self.model.inputs, [self.model.output, self.last_conv_layer.output])
        return iterated_model


def training_model(network, class_count, trainable=False) -> keras.Model:
    """
    loads a pre-trained TensorFlow Keras model
    """
    print("Constructing network")
    networks = {"resnet50": keras.applications.ResNet50V2(weights=None, include_top=False, pooling='avg'),
                "resnet101": keras.applications.ResNet101V2(weights=None, include_top=False, pooling='avg'),
                "resnet152": keras.applications.ResNet152V2(weights=None, include_top=False, pooling='avg'),
                "inception": keras.applications.InceptionV3(weights=None, include_top=False, pooling='avg'),
                "inception_resnet": keras.applications.InceptionResNetV2(weights=None, include_top=False, pooling='avg'),
                "xception": keras.applications.Xception(weights=None, include_top=False, pooling='avg'),
                "nasnet_large": keras.applications.NASNetLarge(weights=None, include_top=False, pooling='avg'),
                "densenet121": keras.applications.DenseNet121(weights=None, include_top=False, pooling='avg'),
                "densenet169": keras.applications.DenseNet169(weights=None, include_top=False, pooling='avg'),
                "densenet201": keras.applications.DenseNet201(weights=None, include_top=False, pooling='avg')
                }

    if network == "inception_resnet":
        num_features = 1536
    elif network == "densenet169":
        num_features = 1664
    elif network == "densenet201":
        num_features = 1920
    elif network == "nasnet_large":
        num_features = 4032
    else:
        num_features = 2048
    feature_extractor_layer = networks[network]

    """ TODO: find a way to create a seed for keraslayer """
    # import the pre-trained network model and leave its parameters untouched
    feature_extractor_layer.trainable = trainable  # change to True if you want to train the pre-trained model

    x = keras.layers.Dense(num_features)(feature_extractor_layer.output)
    x = keras.layers.Dense(class_count, activation='softmax')(x)

    model = keras.Model(inputs=feature_extractor_layer.inputs, outputs=x)

    print("Network constructed")
    return model
