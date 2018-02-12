import numpy as np
import copy


class Network():
    MUTATION_SIZE = 1

    def __init__(self):
        self.layers = []
        self.weights = []
        self.fitness = 0

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

    def mutate(self):
        rand_weights = np.random.choice(self.weights)
        row = np.random.randint(rand_weights.shape[0])
        col = np.random.randint(rand_weights.shape[1])
        rand_weights[row, col] += np.random.uniform(-5, 5)
        if np.random.uniform(0, 1) < 0.1:
            rand_weights[row, col] = 0

    def mutate_safely(self):
        if len(self.layers) != 3:
            raise NotImplementedError

        sensitivity = []

        delta = self.activation_f(self.layers[-1]) * self.activation_f(self.layers[-1], True)
        # print(delta.T)
        # print(self.layers[-2])
        sensitivity.append(delta.T @ self.layers[-2])

        # print(self.weights)
        # print(self.layers)

        delta = delta @ self.weights[-1].T * self.activation_f(self.layers[-2], True)

        sensitivity.append(delta.T @ self.layers[-3])

        mutation_dir_1 = self.MUTATION_SIZE * np.random.uniform(-1, 1, size=sensitivity[0].shape) / sensitivity[0]
        mutation_dir_2 = self.MUTATION_SIZE * np.random.uniform(-1, 1, size=sensitivity[1].shape) / sensitivity[1]

        self.weights[1] += mutation_dir_2
        self.weights[0] += mutation_dir_1

    def copy(self):
        new = Network()
        new.fitness = self.fitness
        new.weights = copy.deepcopy(self.weights)
        new.layers = copy.deepcopy(self.layers)
        return new

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

    def __str__(self):
        return str(self.fitness)


train_x = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]
train_y = [
    (0, 0),
    (1, 1),
    (1, 1),
    (0, 0)
]


def evaluate(n: Network):
    loss = 0
    for x, y in zip(train_x, train_y):
        n.forward_propagate(np.array([x]))
        loss += ((np.array([y]) - n.output) ** 2).sum()
    return loss


def main():
    # np.random.seed(1)

    pop_size = 100
    generations = 100
    elitism = 0.6

    pop = []
    for i in range(pop_size):
        n = Network()
        n.add_layer(2)
        n.add_layer(3)
        n.add_layer(2)
        pop.append(n)

    for gen in range(generations):
        for i, n in enumerate(pop):
            n.fitness = (n.fitness + evaluate(n)) / 2

        pop.sort(key=lambda x: x.fitness)
        pop = pop[:int(len(pop) * (elitism))]

        while len(pop) < pop_size:
            parent = np.random.choice(pop)
            offspr = parent.copy()
            offspr.mutate_safely()
            pop.append(offspr)
    pop.sort(key=lambda x: x.fitness)
    print(pop[0].fitness)


if __name__ == '__main__':
    main()
