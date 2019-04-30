
def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if evaluation and validation_generator:
        if dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone, dataset_type=dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : batch_size,
        'config'           : config,
        'image_min_side'   : image_min_side,
        'image_max_side'   : image_max_side,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            coco_path,
            'train2017',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            coco_path,
            'val2017',
            **common_args
        )
    elif dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            pascal_path,
            'trainval',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            pascal_path,
            'test',
            **common_args
        )
    elif dataset_type == 'csv':
        train_generator = CSVGenerator(
            annotations,
            classes,
            transform_generator=transform_generator,
            **common_args
        )

        if val_annotations:
            validation_generator = CSVGenerator(
                val_annotations,
                classes,
                **common_args
            )
        else:
            validation_generator = None
    elif dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            main_dir,
            subset='train',
            version=version,
            labels_filter=labels_filter,
            annotation_cache_dir=annotation_cache_dir,
            parent_label=parent_label,
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            main_dir,
            subset='validation',
            version=version,
            labels_filter=labels_filter,
            annotation_cache_dir=annotation_cache_dir,
            parent_label=parent_label,
            **common_args
        )
    elif dataset_type == 'kitti':
        train_generator = KittiGenerator(
            kitti_path,
            subset='train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            kitti_path,
            subset='val',
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator
