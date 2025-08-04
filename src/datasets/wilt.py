import os
from pathlib import Path
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.keras import layers, Model, Input, losses


cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'country',
        'income']

cont_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'country', 'income']

def load_adults_data_raw():
    '''Do something about the hard-coded path!
    
    '''
    data_dir = os.path.join(Path(__file__).parent.parent.parent, 'data\\adult_raw\\adult.data')
    adults_df = pd.read_csv(data_dir, names=cols)
    return adults_df

def get_encoding_dict(adults_df):
    '''get mappings from categorical features to numerical and back
    
    '''
    encoding_dict = {}
    decoding_dict = {}
    for c in cat_cols:
        vals = sorted(np.unique(adults_df[c]))
        encoding_dict[c] = dict(zip(vals, np.arange(len(vals)).astype(int)))
        decoding_dict[c] = dict(zip(np.arange(len(vals)).astype(int), vals))
    return encoding_dict, decoding_dict

def normalize_num(data, encoding_dict):
    num_cols = [c for c in data.columns if not c in encoding_dict.keys()]
    Scaler = MinMaxScaler().fit(data[num_cols])
    data[num_cols] = Scaler.transform(data[num_cols])
    return data, Scaler

def load_adults_data(raw_input_data=None):
    '''Load adults dataset and encode the categorical features into numbers
    
    '''
    if raw_input_data is None:
        print("using original adult dataset")
        adults_df = load_adults_data_raw()
    else:
        print("using given synthetic adult dataset")
        adults_df=raw_input_data
    encoding_dict, decoding_dict = get_encoding_dict(adults_df)
    adults_df2 = adults_df.astype(dtype=dict(zip(cont_cols, len(cont_cols)*['float'])))
    for c in cat_cols:
        adults_df2[c] = adults_df2[c].map(encoding_dict[c])
    
    adults_df2, Scaler = normalize_num(adults_df2, encoding_dict)
    return adults_df2, encoding_dict, decoding_dict, Scaler, cont_cols, cat_cols

def one_hot_encoding(data, encoding_dict):
    '''One hot encoding of all categorical columns. Input is a pd.DataFrame and the encoding dict, output a numpy array
    
    col_idxs is an OrderedDict that holds the indices of all original columns in the one-hot encoding
    
    '''
    depths = dict([(k, len(list(encoding_dict[k].keys()))) for k in encoding_dict.keys()])
    cols  = []
    col_idxs = OrderedDict()
    i = 0
    for c in data.columns:
        if c in encoding_dict.keys():
            d = data[c].to_numpy()
            col = tf.one_hot(d, depth=depths[c])
            col_idxs[c] = [i,0]
            i += depths[c]
            col_idxs[c][1] = i
        else:
            col = data[c].to_numpy().reshape((-1,1))
            col_idxs[c] = [i,0]
            i += 1
            col_idxs[c][1] = i
        cols.append(col)
    return np.concatenate(cols, axis=-1), col_idxs

def one_hot_decoding(data, col_idxs, encoding_dict):
    '''Reverse the one hot encoding. 
    Input data is a 2d numpy array, the col_idxs from one_hot_encoding() 
    and the encoding dict, outputs a pd.DataFrame
    
    '''
    cols  = []
    for c,idxs in col_idxs.items():
        if c in encoding_dict.keys():
            d = np.argmax(data[:,idxs[0]:idxs[1]], axis=1, keepdims=True)
        
        else:
            d = data[:,idxs[0]:idxs[1]].reshape((-1,1))
        cols.append(d)
    data2 = np.concatenate(cols, axis=1)
    return pd.DataFrame(data2, columns=list(col_idxs.keys()))
    
    
# class AdultsLoss(losses.Loss):
    
#     def __init__(self, col_idxs, enc_dict, use_logits=True, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
#         super(AdultsLoss, self).__init__(reduction=reduction, **kwargs)
#         self.col_idxs = col_idxs
#         self.enc_dict = enc_dict
#         self.use_logits = use_logits
#         # get all continuous columns
#         self.cat_col_names = list(enc_dict.keys())
#         self.cont_col_names = [k for k in col_idxs.keys() if not k in self.cat_col_names]
#         self.cat_cols = [np.arange(*col_idxs[c]) for c in self.cat_col_names]
#         self.cont_cols = [np.arange(*col_idxs[c]) for c in self.cont_col_names]
#         self.mse = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
#         self.cxes = [losses.CategoricalCrossentropy(from_logits=self.use_logits, reduction=tf.keras.losses.Reduction.NONE) for _ in self.cat_col_names]
        
    
#     def call(self, y_true, y_pred):
#         cont_loss = self.mse(tf.gather(y_true, np.concatenate(self.cont_cols), axis=-1), 
#                              tf.gather(y_pred, np.concatenate(self.cont_cols), axis=-1))
        
#         cat_losses = tf.stack([cxe(tf.gather(y_true, cols, axis=-1), tf.gather(y_pred, cols, axis=-1)) 
#                       for (cxe, cols) in zip(self.cxes, self.cat_cols)], axis=-1)
#         return cont_loss + tf.reduce_sum(cat_losses, axis=-1)

# def make_encoder_adults(data_shape=(110,), latent_dim=10):
#     inx = Input(data_shape)
#     x = layers.Dense(128, activation='relu')(inx)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dense(2*latent_dim, activation=None)(x)
#     return Model(inx, x)

# def make_decoder_adults(col_idxs, enc_dict, data_shape=(110,), latent_dim=10):
#     cont_cols = [c for c in col_idxs if not c in enc_dict.keys()]
#     cont_idxs = np.concatenate([np.arange(*col_idxs[c]) for c in cont_cols])
#     msk = np.zeros(data_shape, dtype=bool)
#     msk[cont_idxs] = 1
#     inx = Input((latent_dim,))
#     x = layers.Dense(128, activation='relu')(inx)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dense(data_shape[0], activation=None)(x)
#     x = tf.where(msk, tf.sigmoid(x), x)
#     return Model(inx, x)

def postprocess(Xraw, col_idxs, enc_dict):
    new_cols = []
    for k,idx in col_idxs.items():
        if k in enc_dict.keys():
            cols = tf.argmax(Xraw[:,idx[0]:idx[1]], axis=1)
            new_col = tf.one_hot(cols, depth=idx[1]-idx[0], axis=1)
        else:
            new_col = Xraw[:,idx[0]:idx[1]]
        new_cols.append(new_col)
    X = tf.concat(new_cols, axis=1)
    return X
    
    


