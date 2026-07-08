"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

from typing import Dict, List, Tuple
import numpy as np
import keras
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES, MADIS_MIN_MAX, MADIS_TOP3_MIN_MAX


def build_model(shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,
                head='maxpool',
                head_units:Tuple[int]=(1024,512),
                use_madis:bool=False,
                madis_min_max=None,
                madis_fusion:str='late'):
    """
    madis_fusion: 'late'  — concatenate MADIS after flattening CNN (original behaviour)
                  'film'  — FiLM-condition CNN feature maps after block 3 (intermediate fusion)
    """
    # Create input layers for each input_variables
    inputs = {}
    for v in input_variables:
        inputs[v]=keras.Input(shape,name=v)
    n_sweeps=shape[2]

    # Create MADIS input if enabled (needs to be created early with other inputs)
    if use_madis:
        _madis_min_max = madis_min_max if madis_min_max is not None else MADIS_MIN_MAX
        madis_input = keras.Input((len(_madis_min_max),), name='madis')
        inputs['madis'] = madis_input

    # Normalize inputs and concate along channel dim
    normalized_inputs=keras.layers.Concatenate(axis=-1,name='Concatenate1')(
        [normalize(inputs[v],v) for v in input_variables]
        )

    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    # Add channel for range folded gates
    if include_range_folded:
        range_folded = keras.Input(shape[:2]+(n_sweeps,),name='range_folded_mask')
        inputs['range_folded_mask']=range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1,name='Concatenate2')(
               [normalized_inputs,range_folded])

    # Input coordinate information
    cin=keras.Input(c_shape,name='coordinates')
    inputs['coordinates']=cin

    x,c = normalized_inputs,cin

    x,c = vgg_block(x,c, filters=start_filters,   ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)   # (60,120)
    x,c = vgg_block(x,c, filters=2*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (30,60)
    x,c = vgg_block(x,c, filters=4*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (15,30)

    # FiLM intermediate fusion: inject MADIS conditioning before block 4
    if use_madis and madis_fusion == 'film':
        film_channels = 4 * start_filters  # channel count at this point
        madis_normalized = normalize_madis(inputs['madis'], _madis_min_max)
        film_mlp = keras.layers.Dense(64, activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(l2_reg),
                                      name='film_dense1')(madis_normalized)
        film_mlp = keras.layers.Dropout(0.3, name='film_dropout')(film_mlp)
        # zeros init → gamma=0, beta=0 at start → identity transform → stable training
        film_params = keras.layers.Dense(2 * film_channels,
                                         activation=None,
                                         kernel_initializer='zeros',
                                         bias_initializer='zeros',
                                         name='film_params')(film_mlp)
        x = FiLMLayer(name='film_conditioning')([x, film_params])

    x,c = vgg_block(x,c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (7,15)

    # Late fusion MADIS branch (original behaviour)
    if use_madis and madis_fusion == 'late':
        madis_normalized = normalize_madis(inputs['madis'], _madis_min_max)
        madis_branch = keras.layers.Dense(32, activation='relu',
                                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                                          name='madis_dense1')(madis_normalized)
        madis_branch = keras.layers.Dropout(0.5, name='madis_dropout1')(madis_branch)
        madis_branch = keras.layers.Dense(16, activation='relu',
                                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                                          name='madis_dense2')(madis_branch)
        madis_branch = keras.layers.Dropout(0.5, name='madis_dropout2')(madis_branch)

    if head=='mlp':
        # MLP head
        x = keras.layers.Flatten()(x)
        # Fuse CNN features with MADIS features (late fusion only — film fusion already happened)
        if use_madis and madis_fusion == 'late':
            x = keras.layers.Concatenate(name='fusion_concatenate')([x, madis_branch])
        x = keras.layers.Dense(units=head_units[0], activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.Dropout(0.4, name='head_dropout1')(x)
        x = keras.layers.Dense(units=head_units[1], activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.Dropout(0.4, name='head_dropout2')(x)
        output = keras.layers.Dense(1)(x)
    elif head=='maxpool':
        # Per gridcell
        x = keras.layers.Conv2D(filters=512, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        # Max in scene
        output = keras.layers.GlobalMaxPooling2D()(x)

    return keras.Model(inputs=inputs,outputs=output)
    
    
    





class FiLMLayer(keras.layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM).

    Takes [features (B, H, W, C), film_params (B, 2C)] and applies:
        output = (1 + gamma) * features + beta
    where gamma and beta are the two halves of film_params, broadcast over (H, W).

    The film_params Dense layer should be initialized to zeros so FiLM starts
    as an identity transform and learns deviations from there.
    """
    def call(self, inputs):
        features, film_params = inputs
        n_channels = features.shape[-1]
        gamma = film_params[:, :n_channels]
        beta  = film_params[:, n_channels:]
        gamma = keras.ops.expand_dims(keras.ops.expand_dims(gamma, axis=1), axis=1)
        beta  = keras.ops.expand_dims(keras.ops.expand_dims(beta,  axis=1), axis=1)
        return (1.0 + gamma) * features + beta

    def get_config(self):
        return super().get_config()


def vgg_block(x,c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu')([x,c])
    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)
    if drop_rate>0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x,c
    
def normalize(x,
              name:str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name]) # [2,]
    n_sweeps=x.shape[-1]
    
    # choose mean,var to get approximate [-1,1] scaling
    var=((min_max[1]-min_max[0])/2)**2 # scalar
    var=np.array(n_sweeps*[var,])    # [n_sweeps,]
    
    offset=(min_max[0]+min_max[1])/2    # scalar
    offset=np.array(n_sweeps*[offset,]) # [n_sweeps,]

    return keras.layers.Normalization(mean=offset,
                                         variance=var,
                                         name='Normalize_%s' % name)(x)

class MadisNormalization(keras.layers.Layer):
    def __init__(self, means, stds, **kwargs):
        super().__init__(**kwargs)
        self.means = list(means)
        self.stds = list(stds)
        self._means = np.array(means, dtype=np.float32)
        self._stds = np.array(stds, dtype=np.float32)

    def call(self, x):
        return (x - self._means) / self._stds

    def get_config(self):
        config = super().get_config()
        config.update({'means': self.means, 'stds': self.stds})
        return config


def normalize_madis(x, madis_min_max=None):
    """Normalize MADIS features to [-1, 1] using MADIS_MIN_MAX ranges."""
    mm = madis_min_max if madis_min_max is not None else MADIS_MIN_MAX
    min_max = np.array(mm, dtype=np.float32)
    means = (min_max[:, 0] + min_max[:, 1]) / 2
    stds  = (min_max[:, 1] - min_max[:, 0]) / 2
    return MadisNormalization(means.tolist(), stds.tolist(), name='Normalize_madis')(x)


