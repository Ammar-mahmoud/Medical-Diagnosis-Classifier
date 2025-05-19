import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = 0
            for j in range(len(neuron['weights'])):
                activation += neuron['weights'][j] * inputs[j]
            activation += neuron['bias']
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate(network, expected):
    num_layers = len(network)
    for i in range(num_layers - 1, -1, -1):
        layer = network[i]
        errors = []
        if i == num_layers - 1:  # output layer
            for j in range(len(layer)):
                neuron = layer[j]
                error = expected[j] - neuron['output']
                errors.append(error)
        else:  # hidden layers
            next_layer = network[i + 1]
            for j in range(len(layer)):
                error = 0
                for neuron in next_layer:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    inputs = row[:-1]
    for i in range(len(network)):
        layer = network[i]
        if i != 0:
            inputs = [n['output'] for n in network[i - 1]]
        for neuron in layer:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['bias'] += l_rate * neuron['delta']

def initialize_network(n_inputs, n_hidden, n_outputs):
    hidden_layer = []
    for _ in range(n_hidden):
        neuron = {
            'weights': [random.uniform(-1, 1) for _ in range(n_inputs)],
            'bias': random.uniform(-1, 1)
        }
        hidden_layer.append(neuron)
    
    output_layer = []
    for _ in range(n_outputs):
        neuron = {
            'weights': [random.uniform(-1, 1) for _ in range(n_hidden)],
            'bias': random.uniform(-1, 1)
        }
        output_layer.append(neuron)

    return [hidden_layer, output_layer]

def train_network(train, n_hidden, n_epoch, l_rate):
    network = initialize_network(len(train[0]) - 1, n_hidden, 1)
    losses = []

    for epoch in range(n_epoch):
        total_loss = 0
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [row[-1]]
            loss = (expected[0] - outputs[0]) ** 2
            total_loss += loss
            backward_propagate(network, expected)
            update_weights(network, row, l_rate)

        losses.append((epoch + 1, total_loss))
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return network, losses

def predict_nn(network, row):
    output = forward_propagate(network, row)
    return 1 if output[0] >= 0.5 else 0
