import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() # need to get the same values within the body 

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU gets rid off all negative numbers hidden layer activation    
class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # responsible   

class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss: 
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len (y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1 )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

dense1 = layer_dense(2,3)
activation1 = activation_ReLU()

dense2 = layer_dense(3,3)
activation2 = activation_softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

#========= first layer using a for loop ================
#layer_outputs = []
# zip --> its a list of lists that adds two list together
# here nueron_weight and nueron_bias are weights and biases but at a certian index
#for nueron_weight, nueron_bias in zip(weights, biases):
#    nueron_output = 0
#   for n_input, weight in zip(inputs, nueron_weight):
#        nueron_output = n_input * weight 
#    nueron_output += nueron_bias
#    layer_outputs.append(nueron_output) 
#print(layer_outputs)

