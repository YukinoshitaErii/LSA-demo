import math
import numpy as np
import pandas as pd
from numpy import float64
import matplotlib.pyplot as plt
import os

# path
DATA_PATH = 'fresh_comp_offline'
USER = 'tianchi_fresh_comp_train_user.csv'
ITEM = 'tianchi_fresh_comp_train_item.csv'
MODEL_PATH = 'trained_model'
OUTPUT_PATH = 'output'
NU_SAMPLE = 15000  # must be even
NU_SAMPLE_TRAIN = 5
MAPPING_PATH = os.path.join(MODEL_PATH, 'mapping.json')


def load(path=DATA_PATH, user=USER):
    # read
    print(f"Reading  {user} ")
    df = pd.read_csv(os.path.join(path, user))
    print("Reading complete")
    print('----------------------------------------------------------------------------------------------------------')
    del df['user_geohash'], df['time']
    print('The basic dataset information: ')
    df.info()
    print('----------------------------------------------------------------------------------------------------------')
    return df


def c_u(df, mapping_path=MAPPING_PATH, model=MAPPING_PATH, nu=NU_SAMPLE, nu_t=NU_SAMPLE_TRAIN):
    # construct category-user matrix
    print('constructing category-user matrix')
    # user,category
    indexI = df.sort_values(by=['user_id', 'item_category'])
    indexI['d'] = indexI['user_id'].duplicated()

    # category index
    indexII = df['item_category'].sort_values().drop_duplicates().reset_index(drop=True)
    index = {'indexi': indexII.values, 'value': indexII.index}
    indexII = pd.DataFrame(index)
    indexII.index = indexII.indexi
    del indexII['indexi'], index, df
    print('index is done')

    print(f'choose sample is {nu}')

    item_user = np.zeros((len(indexII), nu), dtype=int)
    item_userI = np.zeros((len(indexII), int(nu_t)), dtype=int)  # test dataset
    count_user = -1

    for tup in indexI.itertuples():
        if not tup[5]:
            count_user += 1
            if count_user % 100 == 0:
                print(f"{int((count_user / NU_SAMPLE) * 100)}%", end='    ')  # only for test
        if count_user <= nu - 1:
            item_user[indexII.loc[tup[4]], count_user] += 1
        else:
            if count_user <= nu + nu_t - 1:
                item_userI[indexII.loc[tup[4]], count_user - nu] += 1
            else:
                break
    print('\nitem_user matrix have constructed')
    indexII.to_json(model)
    return item_user, item_userI


# method II
def tf_idf(item_user, item_userI, nu=NU_SAMPLE, nu_t=NU_SAMPLE_TRAIN, model=MAPPING_PATH):
    # TF-IDF
    print('----------------------------------------------------------------------------------------------------------')
    print("TF-IDF Processing")
    """"# # method I
    # for c in range(item_user.shape[1]):
    #     print('    {}'.format(c), end='')
    #     for r in range(item_user.shape[0]):
    #         if item_user[r, c] != 0:
    #             # item_user[r, c] = np.true_divide(item_user[r, c], sum(item_user[:, c])) * np.log(
    #             #     np.true_divide(300, sum(item_user[r, :])))
    #             horizon = [np.sum(item_user[i]) for i in range(item_user.shape[0])]
    #             vertical = [np.sum(item_user[:, i]) for i in range(item_user.shape[1])]
    #             item_user[r, c] = (item_user[r, c] / np.sum(item_user[:, c])) * math.log(300 / np.sum(item_user[r, :]))
    #         else:
    #             continue"""

    horizon = [np.sum(item_user[i, :]) for i in range(item_user.shape[0])]  # specific word in all documents
    horizon_I = [np.sum(item_userI[i, :]) for i in range(item_userI.shape[0])]  # specific word in all documents

    vertical = [np.sum(item_user[:, i]) for i in range(item_user.shape[1])]  # number of words in user
    vertical_I = [np.sum(item_userI[:, i]) for i in range(item_userI.shape[1])]  # number of words in user

    print(' index have been calculated ')
    indexII = pd.read_json(model)

    i = np.zeros((len(indexII), nu), dtype=float64)
    ii = np.zeros((len(indexII), int(nu_t)), dtype=float64)
    for c in range(item_user.shape[1]):
        if c % 100 == 0:
            print(f"{int((c / item_user.shape[1]) * 100)}%", end='  ')
        for r in range(item_user.shape[0]):
            if horizon[r] != 0:
                if item_user[r, c] != 0:
                    i[r, c] = (item_user[r, c] / vertical[c]) * math.log(item_user.shape[0] / (horizon[r]))

                else:
                    continue

    # test dataset
    for c in range(item_userI.shape[1]):
        for r in range(item_userI.shape[0]):
            if horizon_I[r] != 0:
                if item_userI[r, c] != 0:
                    ii[r, c] = (item_userI[r, c] / vertical_I[c]) * math.log(item_userI.shape[0] / (horizon_I[r]))
                else:
                    continue

    print('')
    print("TF-IDF")
    return i, ii


def plot_trend(data, output=OUTPUT_PATH):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel='rank', ylabel='nu', title="trend")
    plt.savefig(os.path.join(output, 'trend.jpg'))
    plt.close()


def svd(i, ii):
    # SVD
    print('----------------------------------------------------------------------------------------------------------')
    print('SVD')
    T, s, Vt = np.linalg.svd(i, full_matrices=False)
    plot_trend(s)
    s = s[s > 1000]
    if len(s) > 10: s = s[0:10]
    length = len(s)
    Vt = Vt[0:length, :]
    T = T[:, 0:length]
    II = np.dot(np.dot(ii.T, T), np.linalg.inv(np.diag(s)))

    return length, T, s, Vt, II


# classify
def ranking(A, X):
    a = np.sum(A * X.T, axis=0)
    b_i = np.linalg.norm(A, axis=0)
    b_ii = np.linalg.norm(X)
    a = a / (b_i * b_ii)
    d = pd.DataFrame(a, columns=['III'])
    d = d.sort_values(by=['III'], ascending=False)
    return d


def do_ranking(Vt, II):
    # method I
    print('do ranking')
    d = ranking(Vt, II[0:1, :])
    if d.values[0] >= 0.5:
        Vt[:, d.index[0]]

    else:
        print("this vector does not look quite similar")
    return d, Vt[:, d.index[0]]


# main
def train():
    df = load(DATA_PATH, USER)
    item_user, item_userI = c_u(df)
    i_u = pd.DataFrame(item_user)
    i_u.to_csv('trained_model/i_u.csv')
    # i_u, ii = tf_idf(item_user, item_userI, indexII)
    c, T, s, Vt, II = svd(item_user, item_userI)
    np.save('trained_model/T.npy', T)
    np.save("trained_model/Vt.npy", Vt)
    np.save("trained_model/s.npy", s)
    np.save('trained_model/II.npy', II)
    V = pd.DataFrame(Vt)
    V.to_csv('trained_model/V.csv', index=False, sep=',')
    return item_user, c


def main():
    T = np.load('trained_model/T.npy')
    Vt = np.load('trained_model/Vt.npy')
    s = np.load('trained_model/s.npy')
    II = np.load('trained_model/II.npy')
    d, v = do_ranking(Vt, II)
    return d, v


if __name__ == "__main__":
    train()
