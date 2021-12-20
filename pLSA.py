import numpy as np
np.seterr(divide='ignore',invalid='ignore')

class PLSA(object):

    def __init__(self, c_u, s):
        self.p_w_z = np.random.rand(c_u.shape[0], s)                                # p(w|z)    w-z
        self.p_z_d = np.random.rand(c_u.shape[1], s)                                # p(z|d)    d-z
        self.p_w_d = PLSA.nomalise(c_u).T                                           # P(w|d)    d-w
        self.p_z_d_j = np.zeros([c_u.shape[1], c_u.shape[0], s])                    # P(z|d,w)  d-w-z
        self.c_u = c_u                                                              # W-D

    @staticmethod
    def nomalise(e):
        a = np.sum(e, axis=0)
        return e / a

    def e(self):
        a = self.p_w_z @ self.p_z_d.T
        for k in range(self.p_w_z.shape[1]):  # z
            self.p_z_d_j[:, :, k] = ((np.array([self.p_w_z[:, k]]).T @ np.array([self.p_z_d[:, k].T])) / a).T
            print(f'e{k}', end='  ')

    def m(self):
        for k in range(self.p_w_z.shape[1]):  # z
            a = np.sum(self.c_u.T * self.p_z_d_j[:, :, k])
            self.p_w_z[:, k] = np.sum((self.c_u.T * self.p_z_d_j[:, :, k]), axis=0)/a                           # p(w|z)

            b = np.sum(self.c_u, axis=0)
            self.p_z_d[:, k] = np.sum((self.c_u.T * self.p_z_d_j[:, :, k]), axis=1) / b                         # p(z|d)
            print(f'm{k}', end='  ')


def train(plsa):
    for i in range(200):
        print(f'plsa {i}')
        plsa.e()
        print(f'plsa {i} {np.sum(plsa.p_z_d[:,0])}')
        plsa.m()
        print(f'plsa {i} {np.sum(plsa.p_z_d[:,0])}')
        print('------------------------------------------------------------')
    np.save('p_z_d_j', plsa.p_z_d_j)
    np.save('p_w_z', plsa.p_w_z)
    np.save('p_z_d', plsa.p_z_d)


if __name__ == '__main__':
    pass
