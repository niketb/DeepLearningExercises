import numpy as np
import pandas as pd
import math

#constants
BATCH_SIZE = 100
SOFTMAX_EXPONENT = 2
LEARNING_RATE = 0.01
MAX_ITERATIONS = 10
ERROR_THRESHOLD = 1

#################Initializing Neural Network###################

#initializing weight matrices
w1Matrix = np.random.random((2, 5))
w2Matrix = np.random.random((5, 2))

#initializing the matrices used to update the weight matrices
w1UpdateMatrix = np.zeros((2, 5))
w2UpdateMatrix = np.zeros((5, 2))

#defining a class for the hidden layer nodes
class Node:
    def __init__(self):
        self.fx = 0
        self.dfx = 0

#initializing a hidden layer with 5 nodes
hiddenLayer = []
for i in range(5):
    hiddenLayer.append(Node())

###################################################################

###################Helper functions################################

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(o_x, j):
    denominator = (o_x[0] ** SOFTMAX_EXPONENT) + (o_x[1] ** SOFTMAX_EXPONENT)
    numerator = o_x[j] ** SOFTMAX_EXPONENT
    return numerator / denominator

def readData():
    # Reading data from txt file
    rawData = pd.read_csv("Assignment1Data.txt", sep=" ", header=None)
    rawData.columns = ["x1", "x2", "class"]

    # Applying one-hot encoding on class (0,1)
    dataFrame = pd.get_dummies(rawData, columns=["class"])

    # Storing each row in a dataframe as a list
    processedData = dataFrame.values.tolist()
    return processedData

#applying the W1 weight matrix to the input vector
#the np_ prefix represents that the array variable is in the form of a numpy array
def applyW1(inputMatrix):
    np_input = np.array(inputMatrix)
    return np.matmul(np_input, w1Matrix)

#applying the W2 weight matrix to the output of the forward pass on the hidden layer
def applyW2(hiddenFrwdOut):
    np_hiddenFrwdOut = np.array(hiddenFrwdOut)
    return np.matmul(np_hiddenFrwdOut, w2Matrix)

#applying the transpose of the W2 weight matrix during the backward pass
def applyW2T(np_delta):
    w2T = w2Matrix.transpose()
    #obtaining delta_cap by multiplying delta matrix (i.e., the error obtained at the output)
    #and the transpose of the W2 weight matrix
    np_delta_cap = np.matmul(np_delta, w2T)
    return np_delta_cap

#performing the forward pass for a row of inputs read from the file
def forwardPass(inputFullRow):
    inputMatrix = []

    # since, the row of inputs also contains the one-hot representation,
    # which we don't need at this stage, we only read the 0th and 1st elements
    # to give us a 1 * 2 row vector
    inputMatrix.append(inputFullRow[0])
    inputMatrix.append(inputFullRow[1])

    #computing the input to the hidden layer for the forward pass
    hiddenFrwdIn = applyW1(inputMatrix)

    #initializing the output of the hidden layer for the forward pass
    hiddenFrwdOut = []

    #computing the output of the hidden layer for the forward pass
    #and storing the values of the sigmoid and sigmoid derivative in the node for the backward pass
    for i in range(len(hiddenFrwdIn)):
        hiddenFrwdOut.append(sigmoid(hiddenFrwdIn[i]))
        hiddenLayer[i].fx = hiddenFrwdOut[i]
        hiddenLayer[i].dfx = sigmoidDerivative(hiddenFrwdIn[i])

    #computing the inputs (i.e., x) to the output layer
    o_x = applyW2(hiddenFrwdOut)

    #applying softmax to the the inputs obtained above,
    #which gives us the output of the output layer
    softmaxResult = []
    softmaxResult.append(softmax(o_x, 0))
    softmaxResult.append(softmax(o_x, 1))

    #the target is the one hot representation of the ground truth
    #that is, the 2nd and 3rd elements (where the 0th and 1st elements were the inputs)
    target = []
    target.append(inputFullRow[2])
    target.append(inputFullRow[3])

    #computing the error
    delta = []
    delta.append(softmaxResult[0] - target[0])
    delta.append(softmaxResult[1] - target[1])

    #computing distance from target
    errorMagnitude = math.sqrt(o_x[0] ** 2 + o_x[1] ** 2)

    return o_x[0], o_x[1], errorMagnitude, delta

def backwardPass(inputFullRow, delta):
    global w1UpdateMatrix, w2UpdateMatrix

    inputMatrix = []
    inputMatrix.append(inputFullRow[0])
    inputMatrix.append(inputFullRow[1])

    #first, the input vector, i.e., inputMatrix, is converted to a numpy array
    #by default this array will be a 1 * 2 array (i.e., a row)
    #we need it to be a column vector so we use the reshape function to convert it to a 2 * 1 matrix
    #this transpose operation is indicated by the postfix T in the variable name
    np_input = np.array(inputMatrix)
    np_inputT = np_input.reshape((2, 1))

    #creating a matrix out of the sigmoid values stored in the node
    fxMatrix = []
    for k in range(len(hiddenLayer)):
        fxMatrix.append(hiddenLayer[k].fx)
    np_fxMatrix = np.array(fxMatrix)
    #note that using -1 as an argument in the reshape function means that we're leaving numpy to figure it out
    np_fxMatrixT = np_fxMatrix.reshape((-1, 1))

    np_delta = np.array(delta)
    np_delta = np_delta.reshape((1, 2))

    #the update matrix for each input is the transpose of the output matrix of the hidden layer * delta
    #since we're using hybrid training (mix of online and offline), we don't apply the update directly
    #instead we let it accumulate for a batch (matrix addition) and apply it at the end of the batch
    w2UpdateMatrix = np.add(w2UpdateMatrix, np.matmul(np_fxMatrixT, np_delta))

    #here, we get a 5 * 1 matrix, which needs to be multiplied (dot product) with sigmoid derivative to
    #give the error gradient delta for the next layer
    np_delta_cap = applyW2T(np_delta)

    #the result we got from the previous step was a 5 * 1 array of the shape [[a b c d e]]
    #we wish to convert it to a 1 * 5 array, i.e., [a b c d e]
    #so we simply retain the first element of the array
    np_delta_cap = np_delta_cap[0]

    #the error gradient for W1 will be the scalar product delta * f'(x)
    w1delta = []
    w1delta.append(hiddenLayer[0].dfx * np_delta_cap[0])
    w1delta.append(hiddenLayer[1].dfx * np_delta_cap[1])
    w1delta.append(hiddenLayer[2].dfx * np_delta_cap[2])
    w1delta.append(hiddenLayer[3].dfx * np_delta_cap[3])
    w1delta.append(hiddenLayer[4].dfx * np_delta_cap[4])

    np_w1delta = np.array(w1delta)
    np_w1delta = np_w1delta.reshape((1, 5))

    #adding all the matrices in a batch
    w1UpdateMatrix = np.add(w1UpdateMatrix, np.matmul(np_inputT, np_w1delta))

#updating weights at the end of each batch
def updateWeights():
    global w1Matrix, w2Matrix
    w1Matrix = w1Matrix - (LEARNING_RATE * w1UpdateMatrix)
    w2Matrix = w2Matrix - (LEARNING_RATE * w2UpdateMatrix)

###################################################################

####################"Main" method starts###########################

#reading data from file
data = readData()

#training the model over multiple iterations,
#terminating at MAX_ITERATIONS if error < error threshold condition not reached
for k in range(MAX_ITERATIONS):

    #taking a pass through all records in the dataset
    for i in range(len(data)):
        o1, o2, error, delta = forwardPass(data[i])
        backwardPass(data[i], delta)
        print("x1 = ", data[i][0], "  x2 = ", data[i][1], "  o1(x) = ", o1, "  o2(x) = ", o2, "  err(x) = ", error)

        #update weights at the end of a batch or at the end of the file
        if ((i + 1) % BATCH_SIZE == 0 or i == (len(data) - 1)):
            updateWeights()
            #resetting the update matrices to zero once the weights have been updated
            w2UpdateMatrix = np.zeros((5, 2))
            w2UpdateMatrix = np.zeros((5, 2))

        #at the end of one pass of the file, inform user and check whether error below error threshold was obtained
        #if so, we inform user and exit
        #otherwise, if the max_iterations have been reached, inform user about this
        if (i == (len(data) - 1)):
            print("Round ", k, " done\n")
            if (error < ERROR_THRESHOLD):
                print("Training terminated as error below ERROR_THRESHOLD was obtained.")
                exit(0)
            elif (k == MAX_ITERATIONS - 1):
                print("Training terminated as MAX_ITERATIONS value was reached.")





