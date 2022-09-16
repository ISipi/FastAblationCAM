from tensorflow import keras


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
