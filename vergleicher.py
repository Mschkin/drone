import torch
import cv2
import numpy as np
import itertools
from memory_profiler import profile
import sys
from copy import deepcopy
from pympler import asizeof
from scipy.special import expit
from scipy.signal import convolve
from skimage.measure import block_reduce
import cProfile


def splittimg(f):
    #cv2.imshow('asf', f)
    # cv2.waitKey(1000)
    r = np.zeros((100, 100, 30, 30, 3))
    for i in range(100):
        for j in range(100):
            r[i, j] = f[3 * i:30 + 3 * i, 3 * j:3 * j + 30]
    # print(r.dtype)
    return r / 255


def fuseimg(t):
    r = np.zeros((327, 327, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            r[3 * i:30 + 3 * i, 3 * j:3 * j + 30] = t[i, j]
    cv2.imshow('asf', r)
    cv2.waitKey(0)


def conv(f, I):
    # I: farbe unten rechts
    # f: filterzahl unten rechts farbe
    I = np.swapaxes(np.swapaxes(I, 0, 2), 0, 1)
    c = np.array([convolve(f[i, ::-1, ::-1, ::-1], I, mode='valid')
                  [:, :, 0] for i in range(len(f))])
    # c: filterzahl unten rechts
    return c


cap = cv2.VideoCapture('flasche.mp4')

for i in range(45):
    _, f1 = cap.read()
_, f2 = cap.read()

filter1 = (np.random.rand(3, 6, 6, 3) - 0.5) / 2
filter2 = (np.random.rand(3, 6, 6, 3) - 0.5) / 2
filter3 = (np.random.rand(2, 5, 5, 3) - 0.5) / 2
filter4 = (np.random.rand(4, 4, 4, 2) - 0.5) / 2
filter5 = (np.random.rand(1, 1, 1, 4) - 0.5) / 2
fullyconneted = (np.random.rand(3, 36) - 0.5) / 2

I = np.random.rand(30, 30, 3)


class CompareDescribeClass():
    def __init__(self, filter1, filter2, filter3, filter4, filter5):
        # assumes that I has shape (3,30,30)
        # f tiefe zeile spalte
        # fp tiefe zeile spalte farbe
        self.poolsize = 2
        self.f1p = filter1
        self.f2p = filter2
        self.f3p = filter3
        self.f4p = filter4
        self.f5p = filter5
        self.I_color, self.I_row, self.I_column = (3, 30, 30)
        self.f1p_depth, self.f1p_row, self.f1p_column, self.f1p_color = np.shape(
            self.f1p)
        self.f1_depth, self.f1_row, self.f1_column = self.f1p_depth, self.I_row - \
            self.f1p_row + 1, self.I_column - self.f1p_column + 1
        self.f2p_depth, self.f2p_row, self.f2p_column, self.f2p_color = np.shape(
            self.f2p)
        assert self.f1_depth == self.f2p_color
        self.f2_depth, self.f2_row, self.f2_column = self.f2p_depth, self.f1_row - \
            self.f2p_row + 1, self.f1_column - self.f2p_column + 1
        self.p_depth, self.p_row, self.p_column = self.f2p_depth, self.f2_row // \
            self.poolsize, self.f2_column // self.poolsize
        assert self.f2_row % self.poolsize == 0 and self.f2_column % self.poolsize == 0
        self.f3p_depth, self.f3p_row, self.f3p_column, self.f3p_color = np.shape(
            self.f3p)
        assert self.f2_depth == self.f3p_color
        self.f3_depth, self.f3_row, self.f3_column = self.f3p_depth, self.p_row - \
            self.f3p_row + 1, self.p_column - self.f3p_column + 1
        self.f4p_depth, self.f4p_row, self.f4p_column, self.f4p_color = np.shape(
            self.f4p)
        assert self.f3_depth == self.f4p_color
        self.f4_depth, self.f4_row, self.f4_column = self.f4p_depth, self.f3_row - \
            self.f4p_row + 1, self.f3_column - self.f4p_column + 1
        self.f5p_depth, self.f5p_row, self.f5p_column, self.f5p_color = np.shape(
            self.f5p)
        assert self.f4_depth == self.f5p_color
        self.f5_depth, self.f5_row, self.f5_column = self.f5p_depth, self.f4_row - \
            self.f5p_row + 1, self.f4_column - self.f5p_column + 1

    def __call__(self, I):
        I = np.swapaxes(np.swapaxes(I, 0, 2), 1, 2)
        # make the color first index
        f1 = conv(self.f1p, I)
        sf1 = np.logaddexp(f1, 0)
        # print(np.shape(sf1))
        f2 = conv(self.f2p, sf1)
        # print(np.shape(f2))
        p = block_reduce(f2, (1, self.poolsize, self.poolsize), np.max)
        # print(np.shape(p))
        f3 = conv(self.f3p, p)
        sf3 = np.logaddexp(f3, 0)
        # print(np.shape(sf3))
        f4 = conv(self.f4p, sf3)
        sf4 = np.logaddexp(f4, 0)
        f5 = conv(self.f5p, sf4)
        # print(F)
        sf5 = np.logaddexp(f5, 0)
        return sf5

    def derivatives(self, I):
        I = np.swapaxes(np.swapaxes(I, 0, 2), 1, 2)
        # make the color first index
        f1 = conv(self.f1p, I)
        sf1 = np.logaddexp(f1, 0)
        # print(np.shape(sf1))
        f2 = conv(self.f2p, sf1)
        # print(np.shape(f2))
        p = block_reduce(f2, (1, self.poolsize, self.poolsize), np.max)
        # print(np.shape(p))
        f3 = conv(self.f3p, p)
        sf3 = np.logaddexp(f3, 0)
        # print(np.shape(sf3))
        f4 = conv(self.f4p, sf3)
        sf4 = np.logaddexp(f4, 0)
        f5 = conv(self.f5p, sf4)
        # print(F)
        sf5 = np.logaddexp(f5, 0)
        # debug this it is wrong
        #######################################################################
        back1 = expit(f5)
        print((self.f5p_depth, self.f5p_row, self.f5p_column,
               self.f5p_color, self.f5_depth, self.f5_row, self.f5_column))
        df5 = np.zeros((self.f5p_depth, self.f5p_row, self.f5p_column,
                        self.f5p_color, self.f5_depth, self.f5_row, self.f5_column))
        for (b1, b2, b3, b4, a1, a2, a3), _ in np.ndenumerate(df5):
            df5[b1, b2, b3, b4, a1, a2, a3] = back1[a1, a2, a3] * \
                sf4[b4, a2 + b2, a3 + b3] * (a1 == b1)

        back2 = np.zeros((self.f4_depth, self.f5p_row, self.f5p_column,
                          self.f5_depth, self.f5_row, self.f5_column))
        for (g1, g2, g3, a1, a2, a3), _ in np.ndenumerate(back2):
            back2[g1, g2, g3, a1, a2, a3] = back1[a1, a2, a3] * \
                self.f5p[a1, g2, g3, g1] * expit(f4[g1, g2 + a2, g3 + a3])
        df4 = np.zeros((self.f4p_depth, self.f4p_row,
                        self.f4p_column, self.f4p_color, self.f5_depth, self.f5_row, self.f5_column))
        for (b1, b2, b3, b4, a1, a2, a3), _ in np.ndenumerate(df4):
            df4[b1, b2, b3, b4, a1, a2, a3] = sum(
                [back2[b1, g2, g3, a1, a2, a3] * sf3[b4, g2 + a2 + b2, a3 + g3 + b3] for g2 in range(self.f5p_row) for g3 in range(self.f5p_column)])
        # debug this it is wrong
        """
        back3 = np.zeros((self.f3_depth, self.f3_row,
                          self.f3_column, self.Fp_type))
        for (m1, m2, m3, a), _ in np.ndenumerate(back3):
            back3[m1, m2, m3, a] = sum([back2[e1, e2, e3, a] * self.f4p[e1, m2 - e2, m3 - e3, m1] * expit(
                f3[m1, m2, m3]) for e1 in range(self.f4_depth) for e2 in range(max(0, m2 - self.f4p_row + 1), min(self.f4_row, m2 + 1)) for e3 in range(max(0, m3 - self.f4p_column + 1), min(self.f4_column, m3 + 1))])

        df3 = np.zeros((self.f3p_depth, self.f3p_row,
                        self.f3p_column, self.f3p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df3):
            df3[b1, b2, b3, b4, a] = sum(
                [back3[b1, m2, m3, a] * p[b4, m2 + b2, m3 + b3] for m2 in range(self.f3_row) for m3 in range(self.f3_column)])
        back4 = np.zeros((self.f2_depth, self.f2_row,
                          self.f2_column, self.Fp_type))
        for (x1, x2, x3, a), _ in np.ndenumerate(back4):
            sli = f2[x1, self.poolsize * (x2 // self.poolsize):self.poolsize * (x2 // self.poolsize) + self.poolsize,
                     self.poolsize * (x3 // self.poolsize):self.poolsize * (x3 // self.poolsize) + self.poolsize]
            back4[x1, x2, x3, a] = sum([back3[m1, m2, m3, a] * self.f3p[m1, x2 // self.poolsize - m2, x3 // self.poolsize - m3, x1] *
                                        ((x2 % self.poolsize, x3 % self.poolsize) == np.where(sli == np.max(sli))) for m1 in range(self.f3_depth) for m2 in range(max(0, x2 // self.poolsize - self.f3p_row + 1), min(self.f3_row, x2 // self.poolsize + 1)) for m3 in range(max(0, x3 // self.poolsize - self.f3p_column + 1), min(self.f3_row, x3 // self.poolsize + 1))])

        df2 = np.zeros((self.f2p_depth, self.f2p_row,
                        self.f2p_column, self.f2p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df2):
            df2[b1, b2, b3, b4, a] = sum(
                [back4[b1, x2, x3, a] * sf1[b4, x2 + b2, x3 + b3] for x2 in range(self.f2_row) for x3 in range(self.f2_column)])
        back5 = np.zeros((self.f1_depth, self.f1_row,
                          self.f1_column, self.Fp_type))
        for (g1, g2, g3, a), _ in np.ndenumerate(back5):
            back5[g1, g2, g3, a] = sum([back4[x1, x2, x3, a] * self.f2p[x1, g2 - x2, g3 - x3, g1] * expit(f1[g1, g2, g3])
                                        for x1 in range(self.f2_depth) for x2 in range(max(0, g2 - self.f2p_row + 1), min(self.f2_row, g2 + 1)) for x3 in range(max(0, g3 - self.f2p_column + 1), min(self.f2_row, g3 + 1))])
        df1 = np.zeros((self.f1p_depth, self.f1p_row,
                        self.f1p_column, self.f1p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df1):
            df1[b1, b2, b3, b4, a] = sum(
                [back5[b1, g2, g3, a] * I[b4, g2 + b2, g3 + b3] for g2 in range(self.f1_row) for g3 in range(self.f1_column)])
        """
        return df4


class FinderClass():
    def __init__(self, filter1, filter2, filter3, filter4, fullyconneted):
        # assumes that I has shape (3,30,30)
        # f tiefe zeile spalte
        # fp tiefe zeile spalte farbe
        self.poolsize = 2
        self.f1p = filter1
        self.f2p = filter2
        self.f3p = filter3
        self.f4p = filter4
        self.Fp = fullyconneted
        self.I_color, self.I_row, self.I_column = (3, 30, 30)
        self.f1p_depth, self.f1p_row, self.f1p_column, self.f1p_color = np.shape(
            self.f1p)
        self.f1_depth, self.f1_row, self.f1_column = self.f1p_depth, self.I_row - \
            self.f1p_row + 1, self.I_column - self.f1p_column + 1
        self.f2p_depth, self.f2p_row, self.f2p_column, self.f2p_color = np.shape(
            self.f2p)
        assert self.f1_depth == self.f2p_color
        self.f2_depth, self.f2_row, self.f2_column = self.f2p_depth, self.f1_row - \
            self.f2p_row + 1, self.f1_column - self.f2p_column + 1
        self.p_depth, self.p_row, self.p_column = self.f2p_depth, self.f2_row // \
            self.poolsize, self.f2_column // self.poolsize
        assert self.f2_row % self.poolsize == 0 and self.f2_column % self.poolsize == 0
        self.f3p_depth, self.f3p_row, self.f3p_column, self.f3p_color = np.shape(
            self.f3p)
        assert self.f2_depth == self.f3p_color
        self.f3_depth, self.f3_row, self.f3_column = self.f3p_depth, self.p_row - \
            self.f3p_row + 1, self.p_column - self.f3p_column + 1
        self.f4p_depth, self.f4p_row, self.f4p_column, self.f4p_color = np.shape(
            self.f4p)
        assert self.f3_depth == self.f4p_color
        self.f4_depth, self.f4_row, self.f4_column = self.f4p_depth, self.f3_row - \
            self.f4p_row + 1, self.f3_column - self.f4p_column + 1
        self.Fp_type, self.Fp_color = np.shape(self.Fp)
        assert self.Fp_color == self.f4_depth * self.f4_row * self.f4_column
        self.F_type = self.Fp_type

    def __call__(self, I):
        I = np.swapaxes(np.swapaxes(I, 0, 2), 1, 2)
        # make the color first index
        f1 = conv(self.f1p, I)
        sf1 = np.logaddexp(f1, 0)
        # print(np.shape(sf1))
        f2 = conv(self.f2p, sf1)
        # print(np.shape(f2))
        p = block_reduce(f2, (1, self.poolsize, self.poolsize), np.max)
        # print(np.shape(p))
        f3 = conv(self.f3p, p)
        sf3 = np.logaddexp(f3, 0)
        # print(np.shape(sf3))
        f4 = conv(self.f4p, sf3)
        sf4 = np.logaddexp(f4, 0)
        F = self.Fp@np.reshape(sf4, (self.Fp_color))
        # print(F)
        s = expit(F)
        return s

    def derivatives(self, I):
        I = np.swapaxes(np.swapaxes(I, 0, 2), 1, 2)
        # make the color first index
        f1 = conv(self.f1p, I)
        sf1 = np.logaddexp(f1, 0)
        # print(np.shape(sf1))
        f2 = conv(self.f2p, sf1)
        # print(np.shape(f2))
        p = block_reduce(f2, (1, self.poolsize, self.poolsize), np.max)
        # print(np.shape(p))
        f3 = conv(self.f3p, p)
        sf3 = np.logaddexp(f3, 0)
        # print(np.shape(sf3))
        f4 = conv(self.f4p, sf3)
        sf4 = np.logaddexp(f4, 0)
        F = self.Fp@np.reshape(sf4, (self.Fp_color))
        # print(F)
        s = expit(F)
        back1 = s * (1 - s)
        dF = np.zeros((self.Fp_type, self.Fp_color, self.Fp_type))
        for (j, k, i), _ in np.ndenumerate(dF):
            dF[j, k, i] = back1[i] * (j == i) * \
                np.reshape(sf4, (self.Fp_color))[k]
        back2 = np.zeros((self.f4_depth, self.f4_row,
                          self.f4_column, self.Fp_type))
        for (e1, e2, e3, a), _ in np.ndenumerate(back2):
            back2[e1, e2, e3, a] = back1[a] * self.Fp[a,
                                                      self.f4_row * self.f4_column * e1 + self.f4_column * e2 + e3] * expit(f4[e1, e2, e3])
        df4 = np.zeros((self.f4p_depth, self.f4p_row,
                        self.f4p_column, self.f4p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df4):
            df4[b1, b2, b3, b4, a] = sum(
                [back2[b1, e2, e3, a] * sf3[b4, e2 + b2, e3 + b3] for e2 in range(self.f4_row) for e3 in range(self.f4_column)])
        back3 = np.zeros((self.f3_depth, self.f3_row,
                          self.f3_column, self.Fp_type))
        for (m1, m2, m3, a), _ in np.ndenumerate(back3):
            back3[m1, m2, m3, a] = sum([back2[e1, e2, e3, a] * self.f4p[e1, m2 - e2, m3 - e3, m1] * expit(
                f3[m1, m2, m3]) for e1 in range(self.f4_depth) for e2 in range(max(0, m2 - self.f4p_row + 1), min(self.f4_row, m2 + 1)) for e3 in range(max(0, m3 - self.f4p_column + 1), min(self.f4_column, m3 + 1))])

        df3 = np.zeros((self.f3p_depth, self.f3p_row,
                        self.f3p_column, self.f3p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df3):
            df3[b1, b2, b3, b4, a] = sum(
                [back3[b1, m2, m3, a] * p[b4, m2 + b2, m3 + b3] for m2 in range(self.f3_row) for m3 in range(self.f3_column)])
        back4 = np.zeros((self.f2_depth, self.f2_row,
                          self.f2_column, self.Fp_type))
        for (x1, x2, x3, a), _ in np.ndenumerate(back4):
            sli = f2[x1, self.poolsize * (x2 // self.poolsize):self.poolsize * (x2 // self.poolsize) + self.poolsize,
                     self.poolsize * (x3 // self.poolsize):self.poolsize * (x3 // self.poolsize) + self.poolsize]
            back4[x1, x2, x3, a] = sum([back3[m1, m2, m3, a] * self.f3p[m1, x2 // self.poolsize - m2, x3 // self.poolsize - m3, x1] *
                                        ((x2 % self.poolsize, x3 % self.poolsize) == np.where(sli == np.max(sli))) for m1 in range(self.f3_depth) for m2 in range(max(0, x2 // self.poolsize - self.f3p_row + 1), min(self.f3_row, x2 // self.poolsize + 1)) for m3 in range(max(0, x3 // self.poolsize - self.f3p_column + 1), min(self.f3_row, x3 // self.poolsize + 1))])

        df2 = np.zeros((self.f2p_depth, self.f2p_row,
                        self.f2p_column, self.f2p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df2):
            df2[b1, b2, b3, b4, a] = sum(
                [back4[b1, x2, x3, a] * sf1[b4, x2 + b2, x3 + b3] for x2 in range(self.f2_row) for x3 in range(self.f2_column)])
        back5 = np.zeros((self.f1_depth, self.f1_row,
                          self.f1_column, self.Fp_type))
        for (g1, g2, g3, a), _ in np.ndenumerate(back5):
            back5[g1, g2, g3, a] = sum([back4[x1, x2, x3, a] * self.f2p[x1, g2 - x2, g3 - x3, g1] * expit(f1[g1, g2, g3])
                                        for x1 in range(self.f2_depth) for x2 in range(max(0, g2 - self.f2p_row + 1), min(self.f2_row, g2 + 1)) for x3 in range(max(0, g3 - self.f2p_column + 1), min(self.f2_row, g3 + 1))])
        df1 = np.zeros((self.f1p_depth, self.f1p_row,
                        self.f1p_column, self.f1p_color, self.Fp_type))
        for (b1, b2, b3, b4, a), _ in np.ndenumerate(df1):
            df1[b1, b2, b3, b4, a] = sum(
                [back5[b1, g2, g3, a] * I[b4, g2 + b2, g3 + b3] for g2 in range(self.f1_row) for g3 in range(self.f1_column)])
        return df1


#cv2.imshow('asfa', s[50, 20])
#print(s[10, 10])
# cv2.waitKey(0)


class FindFilterClass:
    def __init__(self, filter1, filter2, filter3, filter4, fullyconneted):
        self.f1 = torch.nn.Conv2d(3, 6, (6, 6), bias=False)
        #print('bla', np.shape(filter1))
        self.f1.weight.data = torch.FloatTensor(
            np.swapaxes(np.swapaxes(filter1, 1, 3), 2, 3))
        self.soft = torch.nn.Softplus()
        # torch.nn.Softplus()
        self.f2 = torch.nn.Conv2d(6, 6, (6, 6), bias=False)
        # print(np.shape(filter2))
        self.f2.weight.data = torch.FloatTensor(
            np.swapaxes(np.swapaxes(filter2, 1, 3), 2, 3))
        # torch.nn.ReLU(),
        self.p = torch.nn.MaxPool2d((2, 2))
        self.f3 = torch.nn.Conv2d(6, 5, (5, 5), bias=False)
        self.f3.weight.data = torch.FloatTensor(
            np.swapaxes(np.swapaxes(filter3, 1, 3), 2, 3))
        # torch.nn.ReLU(),
        self.f4 = torch.nn.Conv2d(5, 4, (4, 4), bias=False)
        self.f4.weight.data = torch.FloatTensor(
            np.swapaxes(np.swapaxes(filter4, 1, 3), 2, 3))
        # torch.nn.ReLU(),
        # torch.nn.View(36),
        self.l = torch.nn.Linear(36, 3, bias=False)
        self.l.weight.data = torch.FloatTensor(fullyconneted)
        self.s = torch.nn.Sigmoid()

    def __call__(self, t):
        t = torch.FloatTensor([np.swapaxes(np.swapaxes(t, 0, 2), 1, 2)])
        t = self.f1(t)
        # print(t)
        t = self.soft(t)
        t = self.f2(t)
        #t = self.r(t)
        t = self.p(t)
        t = self.f3(t)
        t = self.soft(t)
        t = self.f4(t)
        t = self.soft(t)
        t = self.l(t.view(36))
        # print(t)
        t = self.s(t)
        return t


#finder1 = FinderClass(filter1, filter2, filter3, filter4, fullyconneted)
compare1 = CompareDescribeClass(filter1, filter2, filter3, filter4, filter5)
#finder2 = FindFilterClass(filter1, filter2, filter3, filter4, fullyconneted)
# print(finder1(I))
# print(finder2(I))


def finderfunction(I, filter1, filter2, filter3, filter4, fullyconneted):
    finder1 = FinderClass(filter1, filter2, filter3, filter4, fullyconneted)
    return finder1(I)


def comparefunction(I, filter1, filter2, filter3, filter4, filter5):
    finder1 = CompareDescribeClass(filter1, filter2, filter3, filter4, filter5)
    return finder1(I)


def numericdiff(f, inpt, index):
    r = f(*inpt)
    h = 1 / 10000000
    der = []
    for inputnumber, inp in enumerate(inpt):
        if inputnumber != index:
            continue
        ten = np.zeros(tuple(list(np.shape(inp)) +
                             list(np.shape(r))), dtype=np.double)
        for s, val in np.ndenumerate(inp):
            n = deepcopy(inp) * 1.0
            n[s] += h
            ten[s] = (
                f(*(inpt[:inputnumber] + [n] + inpt[inputnumber + 1:])) - r) / h
        der.append(ten)
    return der


d2 = numericdiff(comparefunction, [
    I, filter1, filter2, filter3, filter4, filter5], 4)
d1 = compare1.derivatives(I)
print(np.max(d2 - d1), np.max(d2))
# cProfile.run('finder1.derivatives(I)')
