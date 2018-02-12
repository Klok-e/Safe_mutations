import safe_mutations_pytorch as t

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


def evaluate(n):
    ftn = 0
    for x, y in zip(train_x, train_y):
        pred = n.forward(t.t.FloatTensor([[*x]]))

        prd = pred.data[0][0]
        ftn += (1 / (abs(prd - y[0]) + 1))
    return ftn


def main():
    pop_size = 100
    generations = 100
    elitism = 0.4

    pop = []
    for i in range(pop_size):
        n = t.Network(2)
        n.add_layer(5)
        n.add_layer(10)
        n.add_layer(15)
        n.add_layer(10)
        n.add_layer(5)
        n.add_layer(1)
        pop.append(n)

    for gen in range(generations):
        for i, n in enumerate(pop):
            ftn = evaluate(n)
            if n.fitness is None:
                n.fitness = ftn
            else:
                n.fitness = (n.fitness + ftn) / 2

        pop.sort(key=lambda x: x.fitness, reverse=True)
        pop = pop[:int(len(pop) * elitism)]

        to_add = []
        while len(to_add) < pop_size - len(pop):
            parent = t.random.choice(pop)
            offspr = parent.make_baby()
            to_add.append(offspr)
        pop.extend(to_add)

    # test
    nn = pop[0]
    for i, (x, y) in enumerate(zip(train_x, train_y)):
        print('\n {0} test'.format(i))
        inp = t.t.FloatTensor([[*x]])
        pred = nn.forward(inp)
        print('Error: ', abs(pred.data[0][0] - y[0]))


if __name__ == '__main__':
    main()
