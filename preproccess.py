import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Dot, Activation, Add, Lambda

# tensorflow >= 2.0

from tensorflow.keras import backend as K


# rmse is not implemented by keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# read to a dataframe
df = pd.read_csv('./u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['user_id', 'item_id', 'rating']
df['rating'] = df['rating'].values.astype(np.float32)

user_means = df.groupby(['user_id'])

user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

n_users = len(user_ids)
n_items = len(item_ids)

cached_means = {}

print('Calculating user means')
for usr in user_ids:
    mean = user_means.get_group(usr)['rating'].mean()
    cached_means[usr] = mean

print('Sorting movie for output order')
sorted_item_ids = np.copy(item_ids)
np.sort(sorted_item_ids)
item_id_to_pos = {val: idx for idx, val in enumerate(sorted_item_ids)}

# mean centering for each user
df['rating'] = df.apply(lambda x: x['rating'] - cached_means[x['user_id']], axis=1)

# df.to_pickle('mean_centered_2.pkl')

# df['rating'] = df['rating'].subtract(df['rating'].mean())

min_rating = min(df['rating'])
max_rating = max(df['rating'])

# scale rating for sigmoid output
print('Scaling ratings for sigmoid usage')
rating_scaler = MinMaxScaler()
df['rating'] = rating_scaler.fit_transform(df['rating'].values.reshape(-1, 1))

# df.to_pickle('scaled_sigmoid_2.pkl')

user_enc = OneHotEncoder()

print('Encoding users')
usr_enc_map = dict(zip(user_ids, user_enc.fit_transform(user_ids.reshape(-1, 1))))

X = []
Y = []

print('Preparing X and Y neural input data')
for idx, row in df.iterrows():
    user_id = int(row.user_id)
    item_id = int(row.item_id)
    user_mean = cached_means[user_id]

    rating = row.rating
    user_emb = usr_enc_map[user_id].toarray().squeeze()
    _x = user_emb
    # missing data goes here
    _y = np.full(n_items, user_mean)  # fill with desired missing value / substitute

    _y[item_id_to_pos[item_id]] = rating  # set correct item id rating

    X.append(_x)
    Y.append(_y)

X = np.array(X)
X = np.squeeze(X)
Y = np.array(Y)

print(f"{X.shape, Y.shape}")
# => (100000, 943), (100000, 1682)

# Cross Validation Splits
print("Preparing CV splits")
folds_x = []
valid_x = []

folds_y = []
valid_y = []

chunk_size = len(df) / 5
for i in range(0, 5):
    begin = int(chunk_size * i)
    end = int(chunk_size * (i + 1))
    valid_x.append(X[begin:end])
    folds_x.append(np.concatenate([X[:begin], X[end:]]))

    valid_y.append(Y[begin:end])
    folds_y.append(np.concatenate([Y[:begin], Y[end:]]))


# Training
best_model = None
best_test_loss = None
best_h = None
best_i = None

# for each cv split find best_model
for i in range(0, 5):
    print(f'CV {i}')

    regu = l1(0.1)

    model_in = Input(shape=(n_users,))

    # H = 20
    x = Dense(20, activation='relu', kernel_regularizer=regu, activity_regularizer=regu)(model_in)
    # add more layers here
    model_out = Dense(n_items, kernel_regularizer=regu, activity_regularizer=regu, activation='sigmoid')(x)

    model = Model(inputs=model_in, outputs=model_out)
    # opt = SGD(lr=0.001, momentum=0.2)
    opt = 'adam'

    model.compile(loss=root_mean_squared_error, optimizer=opt, metrics=['accuracy'])
    # early_stop = EarlyStopping(patience=3, monitor='val_loss') for early stopping

    history = model.fit(folds_x[i], folds_y[i], epochs=1, validation_split=0.2)

    # evaluate against cv split
    test_loss = model.evaluate(valid_x[i], valid_y[i])

    if best_test_loss is None or test_loss[0] < best_test_loss:
        best_test_loss = test_loss[0]
        best_model = model
        best_h = history
        best_i = i

print(f'Best model for cv {best_i} with loss {best_test_loss}')

# plot metrics of best split
plt.plot(best_h.history['accuracy'])
plt.plot(best_h.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(best_h.history['loss'])
plt.plot(best_h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
