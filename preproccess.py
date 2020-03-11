import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['user_id', 'item_id', 'rating']
df['rating'] = df['rating'].values.astype(np.float32)
df = df.head(50)

# data preparation
user_means = df.groupby(['user_id'])

to_append = []

cached_means = {}

# takes some 30-60min to run depending on processing power

user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

i = 0
# fill missing movies for each user
for usr in user_ids:
    mean = user_means.get_group(usr)['rating'].mean()
    cached_means[usr] = mean
    i = i + 1
    # print(f'Checking user_id: {usr} - {i / len(user_ids)} % done')
    m = 0
    for item in item_ids:
        if not df[(df['user_id'] == usr) & (df['item_id'] == item)].any().any():
            to_append.append({'user_id': usr, 'item_id': item, 'rating': mean})
            # print(f'    >Filling ${usr}-${item}')
            m = m + 1
    # print(f'${m} items filled')

print(f'Appending {len(to_append)} elements')

df = df.append(to_append)

df.to_pickle('filled.pkl')

# mean centering for each user
df['rating'] = df.apply(lambda x: x['rating'] - cached_means[x['user_id']], axis=1)

df.to_pickle('mean_centered.pkl')


# scale rating for sigmoid output
rating_scaler = MinMaxScaler()
df['rating'] = rating_scaler.fit_transform(df['rating'].values.reshape(-1, 1))

df.to_pickle('scaled_sigmoid.pkl')

# encode user and item ids

# df['user_emb'] = user_enc.fit_transform(df['user_id'].values.reshape(-1, 1)).toarray()
#
# n_users = df['user_id'].nunique()
# n_items = df['item_id'].nunique()
#
# df.to_pickle('with_emb.pkl')
