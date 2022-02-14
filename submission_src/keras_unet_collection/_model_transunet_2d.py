from __future__ import absolute_import

import tensorflow as tf
from tensorflow import range as tf_range
from tensorflow import reshape as tf_reshape
from tensorflow import shape as tf_shape
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Embedding

import utils
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right
from keras_unet_collection.layer_utils import *

# from tensorflow_addons import MultiHeadAttention
from tensorflow.keras.layers import MultiHeadAttention


class ViT_patch_gen(Layer):
    '''
    Split feature maps into patches.
    
    patches = ViT_patch_gen(patch_size)(feature_map)
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)
        
    Output
    ----------
        patches: a three-dimensional tensor of (num_sample*num_patches, patch_size*patch_size)
                 where `num_patches = (width // patch_size) * (height // patch_size)`
                 
    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        
    '''

    def __init__(self, patch_size):
        super(ViT_patch_gen, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf_shape(images)[0]
        patches = extract_patches(images=images,
                                  sizes=[1, self.patch_size, self.patch_size, 1],
                                  strides=[1, self.patch_size, self.patch_size, 1],
                                  rates=[1, 1, 1, 1], padding='VALID', )

        # patches.shape = (num_sample, num_patches, patch_size*patch_size)
        patch_dim = patches.shape[-1]
        patches = tf_reshape(patches, [batch_size, -1, patch_dim])

        # patches.shape = (num_sample*num_patches, patch_size*patch_size)
        return patches


class ViT_embedding(Layer):
    '''
    
    The embedding layer of ViT pathes.
    
    patches_embed = ViT_embedding(num_patches, proj_dim)(pathes)
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        num_patches: number of patches to be embedded.
        proj_dim: number of embedded dimensions. 
        
    Output
    ----------
        embed: Embedded patches.
    
    For further information see: https://keras.io/api/layers/core_layers/embedding/
    
    '''

    def __init__(self, num_patches, proj_dim):
        super(ViT_embedding, self).__init__()
        self.num_patches = num_patches
        self.proj_dim = proj_dim
        self.nums_per_batch = self.num_patches * self.proj_dim
        self.proj = Dense(proj_dim)
        self.pos_embed = Embedding(input_dim=num_patches, output_dim=proj_dim)

    def call(self, patch):
        pos = tf_range(start=0, limit=self.num_patches, delta=1)
        patch_proj = self.proj(patch)
        batch_size = tf.cast(tf.reduce_sum(tf.ones_like(patch_proj)) / self.nums_per_batch, tf.int32)
        pos_embedded = tf.tile(tf.expand_dims(self.pos_embed(pos), 0), [batch_size, 1, 1])
        embed = patch_proj + pos_embedded
        return embed


def ViT_MLP(X, filter_num, activation='GELU', name='MLP'):
    '''
    The MLP block of ViT.
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        X: the input tensor of MLP, i.e., after MSA and skip connections
        filter_num: a list that defines the number of nodes for each MLP layer.
                        For the last MLP layer, its number of node must equal to the dimension of key.
        activation: activation of MLP nodes.
        name: prefix of the created keras layers.
        
    Output
    ----------
        V: output tensor.

    '''
    activation_func = eval(activation)

    for i, f in enumerate(filter_num):
        X = Dense(f, name='{}_dense_{}'.format(name, i))(X)
        X = activation_func(name='{}_activation_{}'.format(name, i))(X)

    return X


def ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT'):
    '''
    
    Vision transformer (ViT) block.
    
    ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT')
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        V: embedded input features.
        num_heads: number of attention heads.
        key_dim: dimension of the attention key (equals to the embeded dimensions).
        filter_num_MLP: a list that defines the number of nodes for each MLP layer.
                        For the last MLP layer, its number of node must equal to the dimension of key.
        activation: activation of MLP nodes.
        name: prefix of the created keras layers.
        
    Output
    ----------
        V: output tensor.
    
    '''
    # Multiheaded self-attention (MSA)
    is_training = False
    if utils.MODE == 'train':
        is_training = True
    V_atten = V  # <--- skip
    V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V_atten)
    V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                 name='{}_atten'.format(name))(V_atten, V_atten, training=is_training)

    # Skip connection
    V_add = add([V_atten, V], name='{}_skip_1'.format(name))  # <--- skip

    # MLP
    V_MLP = V_add  # <--- skip
    V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_MLP)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, activation, name='{}_mlp'.format(name))
    # Skip connection
    V_out = add([V_MLP, V_add], name='{}_skip_2'.format(name))  # <--- skip

    return V_out


def transunet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                      proj_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                      activation='ReLU', mlp_activation='GELU', batch_norm=False, pool=True, unpool=True,
                      name='transunet'):
    '''
    The base of transUNET with an optional ImageNet-trained backbone.
    
    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of ViT) ----------
        proj_dim: number of embedded dimensions.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViTs.
        mlp_activation: activation of MLP nodes.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    X_skip = []
    depth_ = len(filter_num)

    # ----- internal parameters ----- #

    # patch size (fixed to 1-by-1)
    patch_size = 1

    # input tensor size
    input_size = input_tensor.shape[1]

    # encoded feature map size
    encode_size = input_size // 2 ** (depth_ - 2)

    # number of size-1 patches
    num_patches = int(encode_size) ** 2

    # dimension of the attention key (= dimension of embedings)
    key_dim = proj_dim

    # number of MLP nodes
    filter_num_MLP = [num_mlp, proj_dim]

    # ----- UNet-like downsampling ----- #

    X = input_tensor

    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                   batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)
    tap = tf.identity(X)

    # downsampling blocks
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool,
                      batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
        X_skip.append(X)

    # subtrack the last tensor (will be replaced by the ViT output)
    X = X_skip[-1]
    X_skip = X_skip[:-1]

    # ----- ViT block after UNet-like encoding ----- #

    # 1-by-1 linear transformation before entering ViT blocks
    # X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_before'.format(name))(X)

    # feature map to patches
    X = ViT_patch_gen(patch_size)(X_skip[-1])

    # patches to embeddings
    X = ViT_embedding(num_patches, proj_dim)(X)

    # stacked ViTs 
    for i in range(num_transformer):
        X = ViT_block(X, num_heads, key_dim, filter_num_MLP, activation=mlp_activation,
                      name='{}_ViT_{}'.format(name, i))

    # reshape patches to feature maps
    X = tf_reshape(X, (-1, encode_size, encode_size, proj_dim))

    # 1-by-1 linear transformation to adjust the number of channels
    X = Conv2D(filter_num[-1], 3, 2, padding='same', use_bias=False, name='{}_conv_trans_after'.format(name))(X)
    X_skip.append(X)

    v = tf.compat.v1.global_variables()
    n1 = len(v)
    base_params = v

    # ----- UNet-like upsampling ----- #

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        X = UNET_right(X, [X_decode[i], ], filter_num_decode[i], stack_num=stack_num_up, activation=activation,
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))
    v = tf.compat.v1.global_variables()
    upper_params = v[n1:]
    return X, base_params, upper_params, tap


def transunet_2d(input_placeholder, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 proj_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                 activation='ReLU', mlp_activation='GELU', output_activation='Softmax', batch_norm=False, pool=True,
                 unpool=True, name='transunet'):
    '''
    TransUNET with an optional ImageNet-trained bakcbone.
    
    
    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of ViT) ----------
        proj_dim: number of embedded dimensions.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViTs.
        mlp_activation: activation of MLP nodes.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras model.
    
    '''
    X, base_params, upper_params, tap = transunet_2d_base(input_placeholder, filter_num, stack_num_down=stack_num_down,
                                                          stack_num_up=stack_num_up,
                                                          proj_dim=proj_dim, num_mlp=num_mlp, num_heads=num_heads,
                                                          num_transformer=num_transformer,
                                                          activation=activation, mlp_activation=mlp_activation,
                                                          batch_norm=batch_norm, pool=pool,
                                                          unpool=unpool, name=name)
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation,
                      name='{}_outercloudviz_output'.format(name))

    out = tf.concat([tf.keras.layers.Conv2D(1, 1, 1, activation=output_activation,
                                            name='smoothener_outercloudviz_pre')(tap), OUT], axis=-1)
    OUT = tf.keras.layers.Conv2DTranspose(1, 5, 1, activation=None,
                                          name='smoothener_outercloudviz_out')(out)
    OUT = tf.image.resize(OUT, (utils.IM_DIM, utils.IM_DIM), antialias=True, preserve_aspect_ratio=True)
    return OUT