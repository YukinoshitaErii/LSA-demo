import numpy as np


class PLSA(object):

    def __init__(self, c_u, s):
        self.p_w_z = np.random.rand(s, c_u.shape[0])  # p(w|z)    z-w
        self.p_z_d = np.random.rand(c_u.shape[1], s)  # p(z|d)    d-z
        self.p_w_d = PLSA.nomalise(c_u).T              # P(w|d)    d-w
        self.p_z_d_j = np.zeros([c_u.shape[1], c_u.shape[0], s])  # P(z|d,w)  d-w-z
        self.c_u = c_u

    @staticmethod
    def nomalise(e):
        a = np.sum(e, axis=0)
        return e / a

    def e(self):
        for k in range(self.p_w_z.shape[0]):  # z
            for i in range(self.p_w_d.shape[0]):  # document
                for j in range(self.p_w_d.shape[1]):  # word
                    self.p_z_d_j[i, j, k] = (self.p_w_z[k, j] * self.p_z_d[i, k]) / (
                                self.p_w_z[:, j].T @ self.p_z_d[i, :].T)
            print(f'1{k}', end='  ')

    def m(self):
        for k in range(self.p_w_z.shape[0]):  # z
            for j in range(self.p_w_d.shape[1]):    # w
                self.p_w_z[k, j] = (self.c_u[j, :] @ self.p_z_d_j[:, j, k])/(np.sum((self.c_u @ self.p_z_d_j[:, :, k])))       # p(w|z)

            for ii in range(self.p_w_d.shape[0]):  # d
                self.p_z_d[ii, k] = (self.c_u[:, ii].T @ self.p_z_d_j[ii, :, k]) / np.sum(self.c_u[:, ii])    # p(z|d)
            print(f'2{k}', end='  ')
