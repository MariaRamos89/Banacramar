import numpy as np

# First define sigmoid function
def nonlin(value, deriv = False):
    if deriv == True:    #In case we need to use the derivate of the sigmoid
        return value * (1 - value)
    return 1/(1 + np.exp(-value))   # 1 / (1 + e^-x)

# Input data
X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]]);

#Output
y = np.array([[0, 0, 1, 1]]).T

# Seed random (to avoid having the same random numbers)
np.random.seed(1)

# Weight matrices
syn0 = 2 * np.random.random(size=(3, 1)) - 1  #It is good to have a zero mean

#Iterations
for it in range(10000):

    #Layers
    l0 = X #First layer are the inputs
    l1 = nonlin(np.dot(l0, syn0))

    #error
    l1_error = y - l1;

    #parameter to update weights
    l1_delta = l1_error * nonlin(l1, True)

    #updating the weights
    syn0 += np.dot(l0.T, l1_delta)

    if (it % 2000) == 0:
        print(l1_error)


print("Final result: ")
print(l1)
