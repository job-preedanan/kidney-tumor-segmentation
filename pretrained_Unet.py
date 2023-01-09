import tensorflow as tf


def get_pretrained_unet(height, width, channels, output_channels, pretrained='vgg16'):

    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    if pretrained == 'resnet':
        base_model = tf.keras.applications.resnet.ResNet50(input_shape=(height, width, channels),
                                                           include_top=False,
                                                           weights='imagenet')
    elif pretrained == 'vgg16':
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

    base_model.trainable = False
    base_model_output = base_model.get_layer('block5_conv3').output

    # -------------------------------------------DECODER PART ------------------------------------------------------

    enc0 = base_model.get_layer('block1_conv2').output
    enc1 = base_model.get_layer('block2_conv2').output
    enc2 = base_model.get_layer('block3_conv3').output
    enc3 = base_model.get_layer('block4_conv3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                           kernel_size=2,
                                           strides=2,
                                           kernel_initializer='he_uniform')(base_model_output)
    dec3 = tf.keras.layers.BatchNormalization()(dec3)
    dec3 = tf.keras.layers.Concatenate(axis=-1)([dec3, enc3])

    dec2 = decoding_block(first_layer_filter_count * 4, dec3)  # (N/4 x N/4 x 4CH)
    dec2 = tf.keras.layers.Concatenate(axis=-1)([dec2, enc2])

    dec1 = decoding_block(first_layer_filter_count * 2, dec2)  # (N/2 x N/2 x 2CH)
    dec1 = tf.keras.layers.Concatenate(axis=-1)([dec1, enc1])

    dec0 = decoding_block(first_layer_filter_count, dec1)  # (N x N x CH)
    dec0 = tf.keras.layers.Concatenate(axis=-1)([dec0, enc0])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    last = convolution_block(first_layer_filter_count, dec0)
    last = convolution_block(first_layer_filter_count, last)
    last = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(last)
    last = tf.keras.layers.Activation(activation='sigmoid')(last)

    model = tf.keras.Model(inputs=base_model.input, outputs=last)

    return model


def get_pretrained_xnet(height, width, channels, output_channels, pretrained='vgg16'):

    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    if pretrained == 'resnet':
        base_model = tf.keras.applications.resnet.ResNet50(input_shape=(height, width, channels),
                                                           include_top=False,
                                                           weights='imagenet')
    elif pretrained == 'vgg16':
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

    base_model.trainable = False
    base_model_output = base_model.get_layer('block5_conv3').output

    # -------------------------------------------DECODER PART ------------------------------------------------------

    enc0 = base_model.get_layer('block1_conv2').output
    enc1 = base_model.get_layer('block2_conv2').output
    enc2 = base_model.get_layer('block3_conv3').output
    enc3 = base_model.get_layer('block4_conv3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                           kernel_size=2,
                                           strides=2,
                                           kernel_initializer='he_uniform')(base_model_output)
    dec3 = tf.keras.layers.BatchNormalization()(dec3)
    dec3 = tf.keras.layers.Concatenate(axis=-1)([dec3, enc3])

    dec2 = decoding_block(first_layer_filter_count * 4, dec3)  # (N/4 x N/4 x 4CH)
    dec2 = tf.keras.layers.Concatenate(axis=-1)([dec2, enc2])

    dec1 = decoding_block(first_layer_filter_count * 2, dec2)  # (N/2 x N/2 x 2CH)
    dec1 = tf.keras.layers.Concatenate(axis=-1)([dec1, enc1])

    dec0 = decoding_block(first_layer_filter_count, dec1)  # (N x N x CH)
    dec0 = tf.keras.layers.Concatenate(axis=-1)([dec0, enc0])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    last = convolution_block(first_layer_filter_count, dec0)
    last = convolution_block(first_layer_filter_count, last)
    last = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(last)
    last = tf.keras.layers.Activation(activation='sigmoid')(last)

    model = tf.keras.Model(inputs=base_model.input, outputs=last)

    return model