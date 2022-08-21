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


def rouletteWheel (pop, probs):    
    chc = np.random.choice(range(len(pop)), p=probs)
    return pop[chc]


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


def mutation (child):
    for i in range(len(sizes)-1):
        rand = random.randint(sizes[i+1])
        child.biases[i][rand] = np.random.randn()
        rand2 = random.randint(sizes[i+1])
        rand3 = random.randint(sizes[i])
        child.weights[i][rand2][rand3] = np.random.randn()
    
    return child


popsize = 40
genome = [genPop() for _ in range(popsize)]
bests = []
epochs = 10

for _ in epochs:

    fitnesses = []

    for i in genome:
        fitnesses.append(fitness(i))

    bests.append(min(fitnesses))

    invSm = sum([1/k for k in fitnesses])
    probs = [(1/k)/invSm for k in fitnesses]

    selected = [rouletteWheel(genome, probs) for _ in range(popsize)]

    childpop = []

    for i in range(0, popsize, 2):

        par1 = selected[i]
        par2 = selected[i+1]

        children = crossover(par1, par2)

        child1 = mutation(children[0])
        child2 = mutation(children[1])

        childpop.append(child1)
        childpop.append(child2)

    genome = childpop
