import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dot, Embedding, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

n_users = 943
n_items = 1682

df = pd.read_pickle('dataset.pkl')

X = []
Y = []

# do 5-fold cv before sorting movies
for usr in df['user_emb'].unique():
    movies = df[df.loc['user_emb'] == usr]
    movies.sort_values(by='item_id')
    X.append(usr)
    Y.append(movies['rating'].values)

# training

### Inputs Embeddings
user_in = Input(shape=(n_users,))

### Hidden Layers

x = Dense(1024, activation='relu')(user_in)

x = Dense(64, activation='relu')(x)

### Out
out = Dense(n_items, activation='sigmoid')(x)

model = Model(inputs=user_in, outputs=out)
opt = Adam(lr=0.001)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model.compile(loss='mean_squared_error', optimizer=opt)  # mean_absolute_error or

checkpoint = ModelCheckpoint("ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                             monitor='val_loss', save_weights_only=True, save_best_only=True)

early_stop = EarlyStopping(patience=4, monitor='val_loss')

model.fit(X, Y, callbacks=[early_stop, checkpoint])