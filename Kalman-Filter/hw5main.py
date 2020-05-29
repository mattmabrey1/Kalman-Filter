# Kalman filter example demo in Python for 3 different scenarios
# 1 - Trying to estimate the water level of a constant water level in a tank
# 2 - Trying to estimate the water level of a tank that is constantly filling with water
# 3 - Trying to estimate the water level of a tank that is sloshing back and forth in a sin wave function


# Matthew Mabrey
# Dr. Yoon
# Artificial Intelligence CSC 380
# March 23rd 2020
# Homework 5 Extra Credit

import matplotlib.pyplot as plt
import kalmanfilter as kf

plt.rcParams['figure.figsize'] = (10, 8)

# -------------- Initial parameters for Filter 1 (Constant Level)----------------
number_of_iterations = 21              # time steps of kalman filter
f = 0                                   # fill rate of tank
initial_level = 1                       # initial level of water tank
q = 0.0001                              # process variance
r = 0.1                                 # estimate of measurement variance, change to see effect
isLinear = True                         # water level = previous time step water level + fill rate

# initial guesses
initial_xhat_guess = 0
initial_P_guess = 1000

filter1 = kf.KalmanFilter(number_of_iterations, f, initial_level, q, r, initial_xhat_guess, initial_P_guess, isLinear)


plt.figure()
plt.plot(filter1.y_arr, 'r--*', label='measured')
plt.plot(filter1.xhat_arr, 'b-', label='estimated')
plt.plot(filter1.L,color='g',label='true value')
plt.legend()
plt.title('Estimate vs. time step', fontweight='bold')
plt.xlabel('Time Period')
plt.ylabel('Water Level')


# -------------- Initial parameters for Filter 2 (Constant Filling)----------------
number_of_iterations = 21               # time steps of kalman filter
f = 0.1                                 # fill rate of tank
initial_level = 0                       # initial level of water tank
q = 0.00001                             # process variance
r = 0.01                                 # estimate of measurement variance, change to see effect
isLinear = True                         # water level = previous time step water level + fill rate

# initial guesses
initial_xhat_guess = 0
initial_P_guess = 1000


filter2 = kf.KalmanFilter(number_of_iterations, f, initial_level, q, r, initial_xhat_guess, initial_P_guess, isLinear)


plt.figure()
plt.plot(filter2.y_arr, 'r--*', label='measured')
plt.plot(filter2.xhat_arr, 'b-', label='estimated')
plt.plot(filter2.L,color='g',label='true value')
plt.legend()
plt.title('Estimate vs. time step', fontweight='bold')
plt.xlabel('Time Period')
plt.ylabel('Water Level')


# -------------- Initial parameters for Filter 3 (Sin Wave Function)----------------
number_of_iterations = 61               # time steps of kalman filter
f = 0                                   # fill rate of tank
initial_level = 1                       # initial level of water tank
q = 0.00001                             # process variance
r = 0.1                                 # estimate of measurement variance, change to see effect
isLinear = False                        # predefined sin wave function in kalman filter file when isLinear = false

# initial guesses
initial_xhat_guess = 0
initial_P_guess = 1000

filter3 = kf.KalmanFilter(number_of_iterations, f, initial_level, q, r, initial_xhat_guess, initial_P_guess, isLinear)


plt.figure()
plt.plot(filter3.y_arr, 'r--*', label='measured')
plt.plot(filter3.xhat_arr, 'b-', label='estimated')
plt.plot(filter3.L,color='g',label='true value')
plt.legend()
plt.title('Estimate vs. time step', fontweight='bold')
plt.xlabel('Time Period')
plt.ylabel('Water Level')

plt.show()






