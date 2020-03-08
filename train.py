import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import util


n_users = 943
n_items = 1682

orig_df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None, usecols=[0, 1])
orig_df.columns = ['user_id', 'item_id']

df = pd.read_pickle('dataset.pkl')

X = []
Y = []

for usr in df['user_id'].unique():
    movies = df[df.loc['user_id'] == usr]
    movies.sort_values(by='item_id')
    X.append(movies['user_emb'].values)
    Y.append(movies['rating'].values)

folds_x = []
valid_x = []

folds_y = []
valid_y = []

chunk_size = len(orig_df) / 5
for i in range(0, 5):
    begin = chunk_size * i
    end = chunk_size * (i + 1)
    valid_x.append(X[begin:end])
    folds_x.append(X[end:begin])

    valid_y.append(Y[begin:end])
    folds_y.append(Y[end:begin])

# Training
best_model = None
best_test_loss = None
best_h = None
best_i = None

losses = ['rmse', 'mean_absolute_error']
H = [[32], [64], [128], [512], [1024], [128, 64], [512, 64], [1024, 64]]

lr = [0.001, 0.001, 0.05, 0.1]
m = [0.2, 0.6, 0.6, 0.6]
r = [0.1, 0.5, 0.9]

history = []

for i in range(0, 5):
    print(f'CV ${i}')
    for h in H:
        print(f'  h=${h}')
        for loss in losses:
            print(f'  loss->${loss}')

            logdir = "logs/scalars/" + f'cv-{i}_h-${h}_loss-${loss}'
            tensorboard_callback = TensorBoard(log_dir=logdir)

            # Inputs Embeddings
            user_in = Input(shape=(n_users,))

            # Hidden Layers
            x = None
            for _h in h:
                if x is None:
                    x = Dense(_h, activation='relu')(input)
                else:
                    x = Dense(_h, activation='relu')(x)

            # Out
            out = Dense(n_items, activation='sigmoid')(x)

            model = Model(inputs=user_in, outputs=out)
            opt = SGD(lr=0.001)

            l = None
            if loss == 'rmse':
                l = util.root_mean_squared_error
            else:
                l = loss

            model.compile(loss=l, optimizer=opt)  # mean_absolute_error or

            checkpoint = ModelCheckpoint("ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                         monitor='val_loss', save_weights_only=True, save_best_only=True)

            early_stop = EarlyStopping(patience=3, monitor='val_loss')

            hist = model.fit(folds_x[i], folds_y[i], callbacks=[early_stop, checkpoint, tensorboard_callback], epochs=7, validation_split=0.2)
            history.append(hist)

            test_loss = model.evaluate(valid_x[i], valid_y[i])

            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = model
                best_h = h
                best_i = i

print(f'Best model with CV split ${best_i} and test loss: ${best_test_loss}, H = ${best_h}')
