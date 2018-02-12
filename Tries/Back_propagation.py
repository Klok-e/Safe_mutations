import numpy as np


class Network():
    def __init__(self):
        self.layers = []
        self.weights = []

    def add_layer(self, neuron_count):
        if len(self.layers) == 0:
            self.layers.append(np.zeros((1, neuron_count)))
        else:
            curr_layer_arr = np.zeros((1, neuron_count))
            self.layers.append(curr_layer_arr)

            prev_layer_count = self.layers[-2].shape[1]
            random_weights = np.random.random((prev_layer_count, neuron_count)) * 10 - 5  # -5..5
            self.weights.append(random_weights)

    def forward_propagate(self, x):
        np.copyto(self.layers[0], x)
        for index in range(len(self.weights)):
            next_values = self.layers[index] @ self.weights[index]
            if index == len(self.weights) - 1:
                activated = next_values
            else:
                activated = self.activation_f(next_values)
            np.copyto(self.layers[index + 1], activated)

    def backward_propagate(self, y):
        e_out = y - self.layers[-1]
        d_out = 0.01 * e_out * self.activation_f(self.layers[-1], True, 'relu')

        update = self.layers[-2].T @ d_out
        self.weights[-1] += update

        '''
        activations in previous multiply by output deltas and add to weights:
        
        self.layers[-2].T [[-0.2]
                           [ 0.4]]
                           
        d_out  [[ 0.15797132 -0.03370711  0.1583485 ]]
        
        update [[-0.03159426  0.00674142 -0.0316697 ]
                [ 0.06318853 -0.01348285  0.0633394 ]]
        '''

        prev_delta = d_out
        for i in range(-2, -len(self.layers), -1):
            tmp = prev_delta @ self.weights[i + 1].T

            d_layer = tmp * self.activation_f(self.layers[i], True)

            update = self.layers[i - 1].T @ d_layer
            self.weights[i] += update

            prev_delta = d_layer

    @staticmethod
    @np.vectorize
    def activation_f(x, deriv=False, func: str = 'sigmoid'):
        if func == 'sigmoid':
            if deriv is True:
                return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
            return 1 / (1 + np.exp(-x))
        if func == 'relu':
            if deriv is True:
                return 1 if x > 0 else 0.1
            return x if x > 0 else 0.1 * x

    @property
    def output(self):
        return self.layers[-1]


def main():
    # np.random.seed(1)

    n = Network()
    n.add_layer(2)
    n.add_layer(4)
    n.add_layer(1)

    train_x = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]
    train_y = [
        (0,),
        (1,),
        (1,),
        (0,)
    ]
    print('Before training:')
    for x, y in zip(train_x, train_y):
        n.forward_propagate(np.array([x]))
        print(n.output)

    for i in range(10000):
        for x, y in zip(train_x, train_y):
            n.forward_propagate(np.array([x]))
            n.backward_propagate(np.array([y]))

    print('\nAfter training:')
    for x, y in zip(train_x, train_y):
        n.forward_propagate(np.array([x]))
        print(n.output)


if __name__ == '__main__':
    main()
