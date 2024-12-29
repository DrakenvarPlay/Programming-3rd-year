def neuralNetwork(inp, weights):

    prediction = [0,0]

    for i in range(len(weights)):

        prediction[i] = inp * weights[i]

    return prediction



print(neuralNetwork(4, [0.125, 0.125]))

weights = [0,0]
out = neuralNetwork(4, weights)
while out[1] < 0.5:
    while out[0] < 0.5:
        weights[0] += 0.0000001
        out = neuralNetwork(4, weights)
    weights[1] += 0.0001
    out = neuralNetwork(4, weights)
print(weights)