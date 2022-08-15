import numpy as np

def funcZ(w, b, x):
    return np.dot(w, x)+b

def dEdY(ypred, y):
    return -((y/ypred)-((1-y)/(1-ypred)))

def sigdYdX(w, z):
    return (w/((1+np.e**(-z))**2))
    #return y*(1-y)

def sigmoid(z):
    return 1/(1+np.e**(-z))

def dXdW(x):
    return x

def dEdX(ey, yx):
    return ey*yx

def dEdW(ex, x):
    return ex*x

def ffneuron(x, w, b ):
    return sigmoid(funcZ(w, b, x))

def outputneuron(x, w, b, yA):
    y = ffneuron(x, w, b)
    errD = dEdY(y, yA)
    return y, errD


layerNo = 2
inputNo = 4
layer1W = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
layer1B = [0, 0, 0]
layer2W = [[0, 0, 0]]
layer2B = [0]

# inputs is array of arrays of 4
inputs = []

# outputs is array of labels (0 or 1)
outputs = []

ots = []
alpha = 0.2

for k in len(inputs):
    inp = inputs[k]
    l1ots = []
    for j in len(layer1W):
        otp = ffneuron(inp, layer1W[j], layer1B[j])
        l1ots.append(otp)
    l2ots = []
    for j in len(layer2W):
        l2ots = outputneuron(l1ots, layer2W[j], layer2B[j], outputs[k])
    ots = l2ots

    dedy2 = []
    for i in len(layer2W):
        zs = funcZ(layer2W[i], layer2B[i], l1ots)
        for j in len(layer1W):
            dydx = sigdYdX(layer2W[i][j], zs)
            dedx = ots[1]* dydx
            dedw = dedx*l1ots[j]
            layer2W[i][j] = layer2W[i][j] - alpha*dedw

    # note: bias not updates yet
    for j in len(layer1W):
        zs = funcZ(layer1W[i], layer1B[i], inp)
        for j in len(inp):
            dydx = sigdYdX(layer1W[i][j], zs)
            dedx = dedy2[i]*dydx
            dedw = dedx*inp[j]
            layer1W[i][j] = layer1W[i][j] - alpha*dedw

