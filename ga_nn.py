# Neural Network with Genetic Optimization

import random
from random import randint
import numpy as np
import copy
from random import gauss

x = []
y = []

sizes = [40, 20, 10]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    # sizes is a list of the number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a


def genPop ():
    return Network(sizes)


fitnesses = []
def fitness (network):
    errors = []
    for i in range(len(x)):
        pred = network.feedforward(x[i])
        errors.append(pred - y)


def rouletteWheel (pop, fitnesses):
    sm = sum(fitnesses)
    chosen = random.uniform(0, sm)
    curr = 0
    for i in range(len(pop)):
        curr += fitnesses[i]
        if (curr > chosen):
            return pop[i]


def crossover (par1, par2):
    child1 = copy.deepcopy(par1)
    child2 = copy.deepcopy(par2)

    for i in range(len(sizes)-1):
        rand = random.randint(sizes[i+1])
        child1.biases[i][rand::], child2.biases[i][rand::] = child2.biases[i][rand::], child1.biases[i][rand::]
        for j in range(sizes[i+1]):
            rand2 = random.randint(sizes[i])
            child1.weights[i][j][rand2::], child2.weights[i][j][rand2::] = child2.weights[i][j][rand2::], child1.weights[i][j][rand2::]

    return child1, child2
