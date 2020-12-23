import scipy.io
import numpy as np
from numpy import matmul, identity
from matplotlib import pyplot as plt

# load .mat file
icaTest = scipy.io.loadmat('icaTest.mat')
A = icaTest.get('A')
U = icaTest.get('U')

# print shape
n, t, m = U.shape[0], U.shape[1], A.shape[0]
X = matmul(A, U)

# A is 3*3 matrix , U is n*t (3*40) matrix
print("n:", n, " t:", t, " m:", m)

# set a random seed and generate our mixing matrix A
np.random.seed(10)
W = np.random.uniform(0, 0.1, (n, m))
print('initial W:\n', W)

# update algorithm
num = 20000
eta = 0.01
for i in range(num):
    Y = matmul(W, X)
    # print("Y shape:", Y.shape)

    Z = np.zeros((n, t))
    # print("Z.shape:", Z.shape)

    for i in range(n):
        for j in range(t):
            Z[i][j] = 1 / (1 + np.exp(-Y[i][j]))

    Mat = t * identity(n) + matmul(1 - 2 * Z, Y.T)

    delta_W = eta * matmul(Mat, W)
    W = W + delta_W

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
# there are 6*6 entries in correlation matrix, we only print the 3*3 part
# that shows the correlation between original and recovered signals
print('Correlation matrix:\n', r[0:n, n:2*n])

# plot signals
x = np.arange(t)
plt.figure()
for i in range(n):      # original signals
    plt.subplot(n,1,i+1)
    plt.plot(x, sig[i])

plt.figure()
for i in range(n):      # mixed signals
    plt.subplot(n,1,i+1)
    plt.plot(x, X[i])

plt.figure()
for i in range(n):      # recovered signals
    plt.subplot(n,1,i+1)
    plt.plot(x, sig_hat[i])
plt.show()


