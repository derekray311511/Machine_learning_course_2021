from sklearn.metrics import mean_squared_error
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Create Selected Function f(x)
fs = 100                                        # Sampling rate
ts = 1 / fs
T = 10                                          # Period
x = np.linspace(-20, 20, fs*40, endpoint=True)  # Plot range
print('\n======= signal parameters =======')
print('len(x) =', len(x))                       # Num of sample point
square_wave = signal.square(2 * np.pi * x / T)   # quare wave
print('len square wave =', len(square_wave))

# Get power of signal f(x)
plt.rcParams["figure.figsize"] = (8, 5)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(x, square_wave)
plt.title('Signal(square wave)')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')

square_wave_watts = square_wave ** 2
plt.subplot(3, 1, 2)
plt.plot(x, square_wave_watts)
plt.title('Signal Power')
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')

square_wave_db = 10 * np.log10(square_wave_watts)
plt.subplot(3, 1, 3)
plt.plot(x, square_wave_db)
plt.title('Signal Power in dB')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.tight_layout()

# Adding noise using target SNR

# Set a target SNR
target_snr_db = 9
print('Target SNR(db) =', target_snr_db)
print('\n======= learning parameters =======')
# Calculate signal power and convert to dB
sig_avg_watts = np.mean(square_wave_watts)
sig_avg_db = 10 * np.log10(sig_avg_watts)
# Calculate noise according to [2] then convert to watts
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
noise_volts = np.random.normal(
    mean_noise, np.sqrt(noise_avg_watts), len(square_wave_watts))
# Noise up the original signal
square_wave_noise = square_wave + noise_volts
# Plot signal with noise
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, square_wave_noise)
plt.title('Signal with noise')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
# Plot in dB
square_wave_noise_watts = square_wave_noise ** 2
square_wave_noise_db = 10 * np.log10(square_wave_noise_watts)
plt.subplot(2, 1, 2)
plt.plot(x, 10 * np.log10(square_wave_noise**2))
plt.title('Signal with noise (dB)')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.tight_layout()


#======================  Get train Data  ======================#

y = square_wave_noise
data_pair = np.zeros((2, 40*fs))
data_pair[0, :] = x
data_pair[1, :] = y
# data_pair[1, :] = square_wave
# print(data_pair)

# 隨機抽取 train data
# 隨機得到要抽取的 pair 編號
number_of_train_data = 3000
number_choice = random.choices(
    range(0, number_of_train_data), weights=None, cum_weights=None, k=number_of_train_data)

train_data_pair = np.zeros((2, number_of_train_data))
for i in range(number_of_train_data):
    train_data_pair[0, i] = data_pair[0, int(number_choice[i])]
    train_data_pair[1, i] = data_pair[1, int(number_choice[i])]


#======================  Create approximate model  ======================#

def Fourier_Series_Model(x, a, b, T, N):
    """
    Input: 
    x: (1 x n) 1D array input
    N: number of n to approximate Fourier Series
    T: Period
    a: (1 x (N+1)) 1D array coefficient
    b: (1 x N) 1D array coefficient
    =============================================
    Output:
    approximate_func: (1 x n) 1D array output
    """
    approximate_func = np.zeros(len(x))
    for i in range(len(x)):
        approximate_func[i] += a[0]
        for n in range(1, N+1):
            approximate_func[i] += a[n] * \
                np.cos(2*np.pi*n*x[i]/T) + b[n]*np.sin(2*np.pi*n*x[i]/T)

    return(approximate_func)


def learning_cost(predictions, y):
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    m = len(y)
    J = 1 / (2 * m) * np.sum(sqrErrors)

    return J


def batch_learning_cost(predictions, y, M):
    # M: 一次幾批 data
    J = 0
    one_len = len(predictions) / M
    for k in range(M):
        errors = np.subtract(
            predictions[k*one_len:(k+1)*one_len], y[k*one_len:(k+1)*one_len])
        sqrErrors = np.square(errors)
        J += 1 / (2 * M) * np.sum(sqrErrors)

    return J

#======================  Update laws  ======================#


def gradient_descent(X, y, a, b, N, T, alpha, iterations):
    '''
    --------------------------------------
    Input Parameters
    --------------------------------------
    X: (1 x n) 1D array
    y: (1 x n) 1D array
    a: (1 x N+1) 1D array
    b: (1 x N) 1D array
    alpha: learning rate(step size)
    iterations: No(number) of iterations

    Output Parameters
    --------------------------------------
    a, b: final weigths values
    cost_history: Conatins value of cost for each iteration. (iter x N) array.
    --------------------------------------
    '''

    cost_history = np.zeros(iterations)
    a_history = np.zeros((iterations, N+1))
    b_history = np.zeros((iterations, N+1))
    m = len(X)

    for i in range(iterations):
        predictions = Fourier_Series_Model(X, a, b, T, N)
        errors = np.subtract(predictions, y)
        a[0] = a[0] - (alpha / m) * np.sum(errors)
        b[0] = 0
        for n in range(1, N+1):
            a[n] += -(alpha / m) * np.cos(2*np.pi *
                                          n*X/T).dot(errors.transpose())
            b[n] += -(alpha / m) * np.sin(2*np.pi *
                                          n*X/T).dot(errors.transpose())

        cost_history[i] = learning_cost(predictions, y)
        a_history[i, :] = a
        b_history[i, :] = b

    return a, b, cost_history, a_history, b_history


def gradient_descent_with_momentum(X, y, a, b, N, T, alpha, iterations, beta=0.5):
    '''
    --------------------------------------
    Input Parameters
    --------------------------------------
    X: (1 x n) 1D array
    y: (1 x n) 1D array
    a: (1 x N+1) 1D array
    b: (1 x N) 1D array
    alpha: learning rate(step size)
    beta: coefficient of momentum
    iterations: No(number) of iterations

    Output Parameters
    --------------------------------------
    a, b: final weigths values
    cost_history: Conatins value of cost for each iteration. (iter x N) array.
    --------------------------------------
    '''

    cost_history = np.zeros(iterations)
    a_history = np.zeros((iterations, N+1))
    b_history = np.zeros((iterations, N+1))
    m = len(X)
    Vta = np.zeros(N+1)  # step size with momentum(for An)
    Vtb = np.zeros(N+1)  # step size with momentum(for Bn)

    for i in range(iterations):
        predictions = Fourier_Series_Model(X, a, b, T, N)
        errors = np.subtract(predictions, y)
        Vta[0] = beta * Vta[0] - (alpha / m) * np.sum(errors)
        a[0] = a[0] + Vta[0]
        b[0] = 0
        for n in range(1, N+1):
            Vta[n] = beta * Vta[n] - (alpha / m) * np.cos(2*np.pi *
                                                          n*X/T).dot(errors.transpose())
            Vtb[n] = beta * Vtb[n] - (alpha / m) * np.sin(2*np.pi *
                                                          n*X/T).dot(errors.transpose())
            a[n] += Vta[n]
            b[n] += Vtb[n]

        cost_history[i] = learning_cost(predictions, y)
        a_history[i, :] = a
        b_history[i, :] = b

    return a, b, cost_history, a_history, b_history


N_num = 30
iterations = 150
alpha = 0.04
beta = 0.7
print('Number of N =', N_num)
print('iterations =', iterations)
print('step size =', alpha)
print('beta =', beta)
a = np.zeros(N_num+1)
b = np.zeros(N_num+1)
# a, b, cost_history, a_history, b_history = gradient_descent(
#     train_data_pair[0], train_data_pair[1], a, b, N=N_num, T=10, alpha=alpha, iterations=iterations)
a, b, cost_history, a_history, b_history = gradient_descent_with_momentum(
    train_data_pair[0], train_data_pair[1], a, b, N=N_num, T=10, alpha=alpha, iterations=iterations, beta=beta)

predict_wave = Fourier_Series_Model(x, a, b, T=10, N=N_num)
last_cost = cost_history[iterations-1]
print('\nlast cost =', last_cost)

columns_a = []
columns_b = []
# a = a.reshape(1, -1)
# b = b.reshape(1, -1)
for i in range(N_num+1):
    columns_a.append('a'+str(i))
    columns_b.append('b'+str(i))
dataframe_a = pd.DataFrame([a], columns=columns_a)
dataframe_b = pd.DataFrame([b], columns=columns_b)
dataframe_a.to_csv("an.csv")
dataframe_b.to_csv("bn.csv")

#======================  Visualization  ======================#
# Plot the square wave signal
plt.rcParams["figure.figsize"] = (8, 5)
plt.figure()
plt.subplot(211)
plt.plot(x, y)
# Give a title for the square wave plot
plt.title('Sqaure wave - 0.1 Hz sampled at 100 Hz /second')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which='both')
# Provide x axis and line color
plt.axhline(y=0, color='k')
# Set the max and min values for y axis
# plt.ylim(-2, 2)

plt.subplot(212)
plt.plot(x, y, color='b')
plt.plot(x, predict_wave, color='r')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which='both')
# Provide x axis and line color
plt.axhline(y=0, color='k')
plt.tight_layout()

plt.figure()
plt.subplot(311)
plt.plot(range(0, N_num+1), a, 'bo')
plt.plot(range(0, N_num+1), b, 'ro')
plt.xlabel('n')
plt.ylabel('a[n], b[n]')

plt.subplot(312)
plt.plot(range(0, iterations), b_history[:, 1])
plt.xlabel('iteration')
plt.ylabel('a_history[i]')

plt.subplot(313)
plt.plot(range(0, iterations), cost_history)
plt.xlabel('iteration')
plt.ylabel('cost J')
plt.tight_layout()


plt.show()
