# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

# by Andrew D. Straw

# Kalman filter example that was adapted for this specific homework problem
# from a basic Kalman filter implementation designed by Andrew D. Straw

# Matthew Mabrey
# Dr. Yoon
# Artificial Intelligence CSC 380
# March 23rd 2020
# Homework 5 Extra Credit


import numpy as np


class KalmanFilter:
    def __init__(self, n_iter, f, initial_level, q, r, xhat_guess, p_guess, isLinear):

        sz = n_iter                                 # size of array
        y = np.zeros((sz, 2, 1))                    # truth value matrix
        Q = np.array([[q / 3, q / 2], [q / 2, q]])  # process variance matrix

        # allocate space for arrays and arrays of matrices
        xhat = np.zeros((sz, 2, 1))                 # a posteri estimate of x , array of 1x2 matrix
        P = np.zeros((sz, 2, 2))                    # a posteri error estimate ,  array of 2x2 matrix
        xhatminus = np.zeros((sz, 2, 1))            # a priori estimate of x ,  array of 1x2 matrix
        Pminus = np.zeros((sz, 2, 2))               # a priori error estimate , array of 2x2 matrix
        K = np.zeros((sz, 2, 2))                    # gain or blending factor , array of 2x2 matrix
        self.L = np.zeros(sz)                       # water tank level
        F = np.array([[1., 1.], [0., 1.]])
        H = np.array([[1., 0.], [0., 0.]])
        R = np.array([[r, 0.], [0., r]])            # estimate of measurement variance, change to see effect

        # initial guesses
        xhat[0][0] = xhat_guess
        P[0][0][0] = p_guess
        P[0][1][1] = p_guess
        self.L[0] = initial_level

        for k in range(1, n_iter):

            # water level update
            if isLinear:
                self.L[k] = self.L[k - 1] + f
            else:
                self.L[k] = 0.5 * np.sin(0.1 * np.pi * k) + 1

            y[k][0] = np.random.normal(self.L[k], 0.125, 1)

            # time update
            xhatminus[k] = F @ xhat[k - 1]
            Pminus[k] = F @ P[k - 1] @ np.transpose(F) + Q

            # measurement update
            K[k] = Pminus[k] @ np.transpose(H) @ np.linalg.inv((H @ Pminus[k] @ np.transpose(H)) + R)

            xhat[k] = xhatminus[k] + K[k] @ (y[k] - H @ xhatminus[k])

            P[k] = (np.eye(2) - K[k] @ H) @ Pminus[k]

        self.y_arr = np.zeros(sz)
        self.xhat_arr = np.zeros(sz)

        for k in range(0, n_iter):
            self.y_arr[k] = y[k][0]
            self.xhat_arr[k] = xhat[k][0]





