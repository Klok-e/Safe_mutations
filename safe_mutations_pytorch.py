import torch as t
from torch.autograd import Variable
import torch.nn as nn
import random

MUTATION_STRENGTH = 0.1


class Network():
    def __init__(self, input_size):
        self.fitness = None

        self.inp = input_size
        self.weights = []
        self.w_sensitivity = []

        self.activation = nn.LeakyReLU()

        self.bias = 1

    def add_layer(self, size):
        if len(self.weights) == 0:
            self.weights.append(Variable(t.randn(self.inp + 1, size).type(t.FloatTensor), requires_grad=True))
        else:
            self.weights.append(Variable(t.randn(self.weights[-1].shape[1], size).type(t.FloatTensor), requires_grad=True))

    def forward(self, x):
        x = Variable(t.cat((x, t.ones(1, 1) * self.bias), 1))
        y = self.activation.forward(x.mm(self.weights[0]))
        for i in range(1, len(self.weights) - 1):
            y = self.activation.forward(y.mm(self.weights[i]))
        y = y.mm(self.weights[-1])
        output = y.mean()
        output.backward()

        if len(self.w_sensitivity) == 0:
            for i in range(len(self.weights)):
                self.w_sensitivity.append(self.weights[i].grad.data.clone())
        else:
            for i in range(len(self.w_sensitivity)):
                w_sens = self.weights[i].grad.data
                self.w_sensitivity[i] += w_sens

        for w in self.weights:
            w.grad.data.zero_()
        return y

    def make_baby(self):
        offspr = Network(self.inp)
        offspr.fitness = self.fitness * 0.8
        for w in self.weights:
            tmp = w.data.clone()
            # print(tmp)
            offspr.weights.append(Variable(tmp, requires_grad=True))

        for i in range(len(offspr.weights)):
            d_mut = t.rand(offspr.weights[i].shape) * 2 - 1
            res = (d_mut / self.w_sensitivity[i].abs()) * MUTATION_STRENGTH
            offspr.weights[i].data += res
        return offspr
