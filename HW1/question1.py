
#Calculating the value of the specified function at a given point
def calculateFunctionValue(pointCoordinates):
        # Specified function f = 1.5 * x1 ** 2 + x2 ** 2 - 2 * x1 * x2 + 2 * x1 ** 3 + 0.5 * x1 ** 4
        functionVal = 1.5 * pointCoordinates[0] ** 2 + pointCoordinates[1] ** 2 - 2 * pointCoordinates[0] * pointCoordinates[1] + 2 * pointCoordinates[0] ** 3 + 0.5 * pointCoordinates[0] ** 4
        return functionVal

#Calculating partial derivative of the specified function with respect to x1
def ddx1(pointCoordinates):
        ddx1_val = 3 * pointCoordinates[0] - 2 * pointCoordinates[1] + 6 * pointCoordinates[0]**2 + 2 * pointCoordinates[0]**3
        return ddx1_val

#Calculating partial derivative of the specified function with respect to x2
def ddx2(pointCoordinates):
        ddx2_val = 2 * pointCoordinates[1] - 2 * pointCoordinates[0]
        return ddx2_val

#Print all values
def printValues(pointCoordinates, functionValues, x1_derivativeValues, x2_derivativeValues):
        for i, j, k, l in zip(pointCoordinates, functionValues, x1_derivativeValues, x2_derivativeValues):
                print(i[0], i[1], j, k, l)

#Initializing x1, x2, learningRate, errorThreshold
pointCoordinates = [[1, -3]]
errorThreshold = 0.01
learningRate = 0.2
x1_derivativeValues = []
x2_derivativeValues = []

for i in range(0, 1000):
        #Add stopping criteria, i.e, derivative < threshold, then stop
        #gx = ddx(x,y), where gx is gradient
        x1_derivativeValues.append(ddx1(pointCoordinates[i]))
        x2_derivativeValues.append(ddx2(pointCoordinates[i]))
        newPointCoordinates = []
        newPointCoordinates.append(pointCoordinates[i][0] - learningRate * ddx1(pointCoordinates[i]))
        newPointCoordinates.append(pointCoordinates[i][1] - learningRate * ddx2(pointCoordinates[i]))
        pointCoordinates.append(newPointCoordinates)

print('Point values:\n')
#print(pointCoordinates)

functionValues = []

#Storing function values at all points in a list
for list in pointCoordinates:
    functionVal = calculateFunctionValue(list)
    functionValues.append(functionVal)

#Call to print function
printValues(pointCoordinates, functionValues, x1_derivativeValues, x2_derivativeValues)

print("Minimum of the function: ")
print(min(functionValues))

