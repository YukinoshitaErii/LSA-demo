import math
import time
import numpy as np
import pandas as pd
from numpy import float64
import matplotlib.pyplot as plt

# path
PATH = 'fresh_comp_offline'
USER_PATH = 'tianchi_fresh_comp_train_user.csv'
ITEM_PATH = 'tianchi_fresh_comp_train_item.csv'
NU_SAMPLE = 15000  # must be even
NU_SAMPLE_TRAIN = 50
MAPPING_PATH = 'mapping.json'


def load(path=PATH, user=USER_PATH):
    # read
    print(f"Reading  {PATH}\{USER_PATH} ")
    df = pd.read_csv(PATH + "\\" + USER_PATH)
    print("Reading complete")
    localtime = time.asctime(time.localtime(time.time()))
    print('Time is : ', localtime)
    print('---------------------------------------------------------------------------------------------------------')
    del df['user_geohash'], df['time']
    print('The basic dataset information: ')
    df.info()
    print('----------------------------------------------------------------------------------------------------------')
    return df


def c_u(df, mapping_path=MAPPING_PATH):
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

    print(f'choose sample is {NU_SAMPLE}')

    item_user = np.zeros((len(indexII), NU_SAMPLE), dtype=int)
    item_userI = np.zeros((len(indexII), int(NU_SAMPLE_TRAIN)), dtype=int)  # test dataset
    count_user = -1

    for tup in indexI.itertuples():
        if not tup[5]:
            count_user += 1
            if count_user % 100 == 0:
                print(f"{int((count_user/NU_SAMPLE)*100)}%", end='    ')   # only for test
        if count_user <= NU_SAMPLE - 1:
            item_user[indexII.loc[tup[4]], count_user] += 1
        else:
            if count_user <= NU_SAMPLE + NU_SAMPLE_TRAIN - 1:
                item_userI[indexII.loc[tup[4]], count_user - NU_SAMPLE] += 1
            else:
                break
    print('\nitem_user matrix have constructed')
    indexII.to_json(MAPPING_PATH)
    return item_user, item_userI


# method II
def tf_idf(item_user, item_userI):
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
    indexII = pd.read_json(MAPPING_PATH)

    i = np.zeros((len(indexII), NU_SAMPLE), dtype=float64)
    ii = np.zeros((len(indexII), int(NU_SAMPLE_TRAIN)), dtype=float64)
    for c in range(item_user.shape[1]):
        if c % 100 == 0:
            print(f"{int((c/item_user.shape[1])*100)}%", end='  ')
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


def plot_trend(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel='rank', ylabel='nu', title="trend")
    plt.savefig('trend.jpg')
    # plt.show()
    # plt.ion()
    # plt.pause(15)  # 显示秒数
    # plt.close('all')


def svd(i):
    # SVD
    print('----------------------------------------------------------------------------------------------------------')
    print('SVD')
    T, s, Vt = np.linalg.svd(i, full_matrices=False)

    # rescale
    n = 0
    for i in range(len(s)):
        a = s[i+2] - s[i]/2
        print(f'a {i} is {a}')
        n += 1
        if 0 < a < 219:
            break

    plot_trend(s)
    s = s[s > s[n-1]]
    length = len(s)
    Vt = Vt[0:length, :]
    T = T[:, 0:length]

    II = np.dot(np.dot(i.T, T), np.linalg.inv(np.diag(s)))
    # return T, s, Vt, II

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
    return Vt[:, d.index[0]]


# main
def main():
    df = load(PATH, USER_PATH)
    item_user, item_userI, indexII = c_u(df)
    i, ii = tf_idf(item_user, item_userI, indexII)
    T, s, Vt, II = svd(i)
    # v = do_ranking(Vt, II)


if __name__ == "__main__":
    main()
