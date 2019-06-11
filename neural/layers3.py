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
y = np.array([[0], 
              [1], 
              [1], 
              [0]])

# Seed random (to avoid having the same random numbers)
np.random.seed(1)

# Weight matrices
syn0 = 2 * np.random.random(size=(3, 4)) - 1  #It is good to have a zero mean
syn1 = 2 * np.random.random(size=(4 ,1)) - 1 

#Iterations
for it in range(40000):

    #Layers
    l0 = X #First layer are the inputs
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    #error
    l2_error = y - l2

    #parameter to update weights
    l2_delta = l2_error * nonlin(l2, True)

    #l1 error afecting second layer
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, True)

    #updating the weights => Gradient descent
    syn0 += np.dot(l0.T, l1_delta)
    syn1 += np.dot(l1.T, l2_delta)

    if (it % 2000) == 0:
        print(l2_error)


print("Final result: ")
print(l2)
