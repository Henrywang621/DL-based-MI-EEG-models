from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate


def EEGNet_fusion(nb_classes, Chans=64, Samples=80,
                  dropoutRate=0.5, norm_rate=0.25, dropoutType='Dropout', cpu=False):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (64, 1)
        conv_filters2 = (96, 1)
        conv_filters3 = (128, 1)

        depth_filters = (1, Chans)
        pool_size = (4, 1)
        pool_size2 = (8, 1)
        separable_filters = (8, 1)
        separable_filters2 = (16, 1)
        separable_filters3 = (32, 1)

        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 64)
        conv_filters2 = (1, 96)
        conv_filters3 = (1, 128)

        depth_filters = (Chans, 1)
        pool_size = (1, 4)
        pool_size2 = (1, 8)
        separable_filters = (1, 8)
        separable_filters2 = (1, 16)
        separable_filters3 = (1, 32)

        axis = 1

    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters,
                             use_bias=False, padding='same')(block1)  # 8
    block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = Flatten()(block2)  # 13

    # 8 - 13

    input2 = Input(shape=input_shape)
    block3 = Conv2D(F1_2, conv_filters2, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input2)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D2,
                             depthwise_constraint=max_norm(1.))(block3)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D(pool_size)(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2_2, separable_filters2,
                             use_bias=False, padding='same')(block3)  # 22
    block4 = BatchNormalization(axis=axis)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D(pool_size2)(block4)
    block4 = dropoutType(dropoutRate)(block4)
    block4 = Flatten()(block4)  # 27
    # 22 - 27

    input3 = Input(shape=input_shape)
    block5 = Conv2D(F1_3, conv_filters3, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input3)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D3,
                             depthwise_constraint=max_norm(1.))(block5)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D(pool_size)(block5)
    block5 = dropoutType(dropoutRate)(block5)

    block6 = SeparableConv2D(F2_3, separable_filters3,
                             use_bias=False, padding='same')(block5)  # 36
    block6 = BatchNormalization(axis=axis)(block6)
    block6 = Activation('elu')(block6)
    block6 = AveragePooling2D(pool_size2)(block6)
    block6 = dropoutType(dropoutRate)(block6)
    block6 = Flatten()(block6)  # 41

    # 36 - 41

    merge_one = concatenate([block2, block4])
    merge_two = concatenate([merge_one, block6])

    flatten = Flatten()(merge_two)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=[input1, input2, input3], outputs=softmax)