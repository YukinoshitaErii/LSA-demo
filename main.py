import process_data as pre
import pLSA

if __name__ == '__main__':
    df = pre.load()
    item_user, item_userI = pre.c_u(df)
    # i, ii = pre.tf_idf(item_user, item_userI)
    c, T, s, Vt, II = pre.svd(item_user)
    print(c)
    plsa = pLSA.PLSA(item_user, c)

    for i in range(200):
        print(f'plsa {i}')
        plsa.e()
        print(f'plsa {i} {plsa.p_z_d[0,0]}')
        plsa.m()
        print(f'plsa {i} {plsa.p_z_d[0,0]}')
        print('------------------------------------------------------------')

