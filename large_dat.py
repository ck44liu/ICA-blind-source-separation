from scipy.io import loadmat  # import loadmat to read .mat file
# from scipy.io import wavfile  # import wavfile if we want to transform to sound files
import numpy as np
from numpy import array, matmul, identity, longdouble
from matplotlib import pyplot as plt

# use loadmat to read sounds.mat
U = loadmat('sounds.mat').get('sounds')
U = array(U, dtype=longdouble)
(n, t) = U.shape
print('U shape: {}'.format((n,t)))  # should be n=5 and t=44000

# set m, the number of mixed signals we want to have
m = 5
# set a random seed and generate our mixing matrix A
np.random.seed(10)
A = np.random.uniform(-5, 5, (m, n))
print('A:\n', A)

# mix the signals
X = matmul(A, U)

# initialize matrix W with small random values
W = np.random.uniform(0, 1e-3, (n, m))
print('initial W:\n', W)

# update algorithm
num = 100
eta = 1e-5
for k in range(num):
    Y = matmul(W, X)
    # print("Y shape:", Y.shape)

    Z = np.zeros((n, t), dtype=longdouble)
    # print("Z.shape:", Z.shape)

    # compute Z[i][j], note that if Y[i][j] is too large or too small,
    # the exponent will be 0 or very large; to make the computation faster,
    # we directly set Z[i][j] to 1 or 0 in such case.
    for i in range(n):
        for j in range(t):
            if Y[i][j] > 20:     # exp will be close to 0
                Z[i][j] = 1
            elif Y[i][j] < -20:  # exp will be very large
                Z[i][j] = 0
            else:
                Z[i][j] = 1 / (1 + np.exp(-Y[i][j]))

    Mat = t * identity(n) + matmul(1 - 2 * Z, Y.T)

    delta_W = eta * matmul(Mat, W)
    W = W + delta_W

    # since the algorithm takes longer time to run than the small dataset one,
    # we print the W after each 20 iterations to check the progress
    if k % 20 == 0:
        print('W after iteration', k, ':\n', W)

print('W after', num, 'iterations:\n', W)

# compute our estimation of U
U_hat = matmul(W, X)

# create sig and sig_hat to store original and recovered signals
sig = np.zeros((n,t))
sig_hat = np.zeros((n,t))

for i in range(n):
    sig[i,:] = U[i, :]
    sig_hat[i,:] = U_hat[i, :]
    # rescale the recovered sig_hat to [0,1]
    # sig_hat[i,:] = (sig_hat[i,:]-min(sig_hat[i,:]))/(max(sig_hat[i,:])-min(sig_hat[i,:]))

# compute and print correlation matrix
r = np.corrcoef(sig, sig_hat)
# there are 10*10 entries in correlation matrix, we only print the 5*5 part
# that shows the correlation between original and recovered signals
print('Correlation matrix:\n', r[0:n, n:2*n])

# plot signals
x = np.arange(t)
plt.figure()
for i in range(n):      # original signals
    plt.subplot(n,1,i+1)
    plt.plot(x, sig[i])

plt.figure()
for i in range(m):      # mixed signals
    plt.subplot(m,1,i+1)
    plt.plot(x, X[i])

plt.figure()
for i in range(n):      # recovered signals
    plt.subplot(n,1,i+1)
    plt.plot(x, sig_hat[i])
plt.show()

# # use wavfile to generate the .wav sound files, scaling might be needed to
# # ensure the audibility of the sound files
# wavfile.write('original_sounds_1.wav', 11000, np.int16(sig[0,:]*40000))
# wavfile.write('mixed_sounds_1.wav', 11000, np.int16(mix[0,:]*40000))
# wavfile.write('recovered_sounds_1.wav', 11000, np.int16(sig_hat[0,:]*40000))