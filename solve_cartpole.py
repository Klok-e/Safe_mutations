import safe_mutations_pytorch as t
import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
observation, reward, done, info = env.step(env.action_space.sample())


def evaluate(n, render=False):
    global observation

    ftn = 0
    env.reset()
    done = False
    while not done:
        if render:
            #env.render()
            pass
        obs = t.t.FloatTensor([observation])
        pred = n.forward(obs)
        action = np.argmax(pred.data.numpy())
        observation, reward, done, info = env.step(action)

        ftn += reward
    return ftn


def main():
    pop_size = 100
    generations = 15
    elitism = 0.4

    pop = []
    for i in range(pop_size):
        n = t.Network(4)
        n.add_layer(50)
        n.add_layer(500)
        n.add_layer(50)
        n.add_layer(2)
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

        print(max(pop, key=lambda x: x.fitness).fitness)

    # test
    nn = pop[0]
    evaluate(nn, render=True)


if __name__ == '__main__':
    main()
