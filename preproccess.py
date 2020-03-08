import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['user_id', 'item_id', 'rating']
df['rating'] = df['rating'].values.astype(np.float32)
# df = df.head(100) # quicker testing-eval

# data preparation
user_means = df.groupby(['user_id'])

to_append = []

cached_means = {}

# takes some 30-60min to run depending on processing power

# fill missing movies for each user
for usr in df['user_id'].unique():
    mean = user_means.get_group(usr)['rating'].mean()
    cached_means[usr] = mean

    print(f'Checking user_id: ${usr}')
    for item in df['item_id'].unique():
        if df[(df['user_id'] == usr) & (df['item_id'] == item)].any().any():
            to_append.append([usr, item, mean])
            print(f'    >Filling ${usr}-${item}')

print(f'Appending {len(to_append)} elements')

df = df.append(to_append)

# mean centering for each user
df['rating'] = df.apply(lambda x: x - cached_means[x['user_id']])

# scale rating for sigmoid output
rating_scaler = MinMaxScaler()
df['rating'] = rating_scaler.fit_transform(df['rating'].values.reshape(-1, 1))

# encode user and item ids
user_enc = OneHotEncoder()
df['user_emb'] = user_enc.fit_transform(df['user_id'].values.reshape(-1, 1))

n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()

df.to_pickle('dataset.pkl')
