from tensorflow import keras
from tensorflow import train
from os import listdir
from typing import Any


def training_model(network: str, class_count: int, trainable: bool = False) -> keras.Model:
    """
    loads a pre-trained TensorFlow Keras model
    """

    print("Constructing network")
    networks = {"resnet50": keras.applications.ResNet50V2(weights=None, include_top=False, pooling='avg'),
                "resnet101": keras.applications.ResNet101V2(weights=None, include_top=False, pooling='avg'),
                "resnet152": keras.applications.ResNet152V2(weights=None, include_top=False, pooling='avg'),
                "inception": keras.applications.InceptionV3(weights=None, include_top=False, pooling='avg'),
                "inception_resnet": keras.applications.InceptionResNetV2(weights=None, include_top=False,
                                                                         pooling='avg'),
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

    # import the pre-trained network model and leave its parameters untouched
    feature_extractor_layer.trainable = trainable  # change to True if you want to train the pre-trained model

    x = keras.layers.Dense(num_features)(feature_extractor_layer.output)
    x = keras.layers.Dense(class_count, activation='softmax')(x)

    model = keras.Model(inputs=feature_extractor_layer.inputs, outputs=x)

    print("Network constructed")
    return model


class PrepareModel:
    def __init__(self, model: Any = None):
        self.model = model

        # load_checkpoint:
        self.status = None

        # create_iterated_model:
        self.select_conv_layer = None
        self.conv_layers = None

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
        checkpoint = train.Checkpoint(optimizer=optimizer, model=self.model)
        manager = train.CheckpointManager(checkpoint,
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
        self.model = keras.Model(self.model.inputs, [self.model.output, last_conv_layer.output])

    def get_model(self):
        """ return the model """
        return self.model

    def get_conv_data(self):
        """ return the list of convolutional layers and the selected (final) convolutional layer """
        return self.conv_layers, self.select_conv_layer
