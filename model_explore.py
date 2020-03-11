import util
import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import tensorflow.keras as keras

hidden_activation = 'relu'

user_in = Input(shape=(n_users,))
x = Dense(32, activation=hidden_activation, kernel_initializer=keras.initializers.glorot_normal(seed=None), bias_initializer='zeros')(user_in)

out = Dense(n_items, activation='sigmoid')(x)

model = Model(inputs=user_in, outputs=out)
opt = SGD(lr=0.001)

mon = 'val_acc'

model.compile(loss='mse', optimizer=opt, metrics=['mae', 'acc'])  # mean_absolute_error or

hist = model.fit(X, Y, epochs=7, validation_split=0.2)
