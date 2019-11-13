from network import *
import os

"""Use this as a general framework to use the Network class."""


def newNetwork(layers):
    net = Network(layers)
    with open('network_info.txt', 'w') as n:
        n.write(str(net.sizes))
    with open('weights.txt', 'w') as w:
        w.write(str(net.weights))
    with open('biases.txt', 'w') as b:
        b.write(str(net.biases))
    return net


def loadNetwork():
    n = open('network_info.txt', 'r')
    w = open('weights.txt', 'r')
    b = open('biases.txt', 'r')
    info = []
    for line in n.readlines():
        info.append(line)
    print(info)
    net = Network(info[0])
    net.biases = b
    net.weights = w
    return net


def saveNetwork(net):
    os.remove('network_info.txt')
    os.remove('weights.txt')
    os.remove('biases.txt')
    with open('network_info.txt', 'w') as n:
        n.write(net.num_layers + '/n' + net.sizes)
    with open('weights.txt', 'w') as w:
        w.write(net.weights)
    with open('biases.txt', 'w') as b:
        b.write(net.biases)


def getNetwork():
    try:
        return loadNetwork()
    except:
        layers = [6, 6]  # input, 2 hidden, output, 6 nodes in each
        return newNetwork(layers)


def trainNetwork(net, inputs=None):
    try:
        with open('training_data.txt', 'r') as training_data:
            train_data = []
            for line in training_data:
                train_data.append(line)

            ' how many times you want to use this data to train the network '
            ' be careful not to over train on one data set '
            iterations = 5
            mini_batch_size = len(train_data) / iterations  # subjective
            rate = 1
            net.SGD(train_data, iterations, mini_batch_size, rate, inputs)
    except:
        print("Input proper Training Data")


def main():
    net = getNetwork()
    for i in range(1):
        inputs = [0.25,
                  0.73,
                  0.11,
                  0.55,
                  0.96,
                  0.75]
        '''insert lines to get input data'''

        '''inputs = None  # change to actually get something
        trainNetwork(net, inputs)  # remove inputs to speed up training, leave inputs to see progress of training'''
        output_node_index = net.get_output_index(inputs)
        'print(output_node_index)'


if __name__ == '__main__':
    main()
