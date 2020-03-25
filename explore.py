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

df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['user_id', 'item_id', 'rating']
df['rating'] = df['rating'].values.astype(np.float32)
# df = df.head(100) # quicker testing-eval

# data preparation
user_means = df.groupby(['user_id'])
# ['rating'].transform('mean')
# fill missing values by some transform
# df['rating'] = df['rating'].fillna(user_means)

to_append = []

# fill missing movies for each user
for usr in df['user_id'].unique():
    mean = user_means.get_group(usr)['rating'].mean()
    print(f'Checking user_id: ${usr}')
    for item in df['item_id'].unique():
        if df[(df['user_id'] == usr) & (df['item_id'] == item)].any().any():
            to_append.append([usr, item, mean])
            print(f'    >Filling ${usr}-${item}')

print(f'Appending {len(to_append)} elements')

df = df.append(to_append)

df.to_pickle('dataset.pkl')

# mean centering for each user
df['rating'] = df['rating'].subtract(user_means)
# scale rating for sigmoid output
rating_scaler = MinMaxScaler()
df['rating'] = rating_scaler.fit_transform(df['rating'].values.reshape(-1, 1))

# encode user and item ids
user_enc = OneHotEncoder()
df['user_emb'] = user_enc.fit_transform(df['user_id'].values.reshape(-1, 1))

n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()

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
