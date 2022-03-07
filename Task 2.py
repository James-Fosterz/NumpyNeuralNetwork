from keras.utils.np_utils import to_categorical
import numpy as np
import time
from mlxtend.data import loadlocal_mnist


# Loading the dataset
x_train, y_training = loadlocal_mnist(
    images_path='Dataset/train-images.idx3-ubyte', 
    labels_path='Dataset/train-labels.idx1-ubyte'
    )
  
x_val, y_validate = loadlocal_mnist(
    images_path='Dataset/t10k-images.idx3-ubyte', 
    labels_path='Dataset/t10k-labels.idx1-ubyte'
    )

# Adjusts data for use in neural network including converting labels to one hot vectors 
x_train = np.array(x_train).astype(np.float32)
y_train = to_categorical(y_training)
x_val = np.array(x_val).astype(np.float32)
y_val = to_categorical(y_validate)

# Normalisation of training and testing data
for img in range(len(x_train)):
        x_train[img]= x_train[img]/255
for img in range(len(x_val)):
        x_val[img] = x_val[img]/255


neuralnetwork = [
    {"_input": 784, "_output": 100, "activation": "relu"},
    {"_input": 100, "_output": 50, "activation": "relu"},
    {"_input": 50, "_output": 25, "activation": "sigmoid"},
    {"_input": 25, "_output": 10, "activation": "softmax"}
    ]



# Neural network class
class DeepNeuralNetwork():
    def __init__(self, neuralnetwork, epochs=60, l_rate=0.2): 
        self.epochs = epochs
        self.l_rate = l_rate
        self.NN = neuralnetwork
        self.params = self.initialization()

    
    # Initisalisation of parameters dictionary
    def initialization(self):       
        params = {} 
        
        # Generates correct number of weights and biases for each layer depending on chosen archetecture and puts them in a dictionary
        for index, layer in enumerate(self.NN):
            num = index + 1
            params['W' + str(num)] = np.random.randn(self.NN[index]["_output"], self.NN[index]["_input"]) * np.sqrt(1. / self.NN[index]["_output"])
            params['B' + str(num)] = np.random.randn(self.NN[index]["_output"]) * np.sqrt(1. / self.NN[index]["_output"])
        
        return params
    
    
    # Activation functions used for forward and backward propgation as well as outputs
    # Sigmoid function and its derivative
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    # Relu function and its derivative
    def relu(self, x, derivative=False):
        if derivative:
            x[x > 0] = 1
            x[x <= 0] = 0
            return x
        return np.maximum(0,x)
    
    # Softmax function and its derivative
    def softmax(self, x, derivative=False):
        exps = np.exp(x - np.max(x))
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    
    # Cross entropy function and its derivative 
    def cross_entropy(self, x, y, derivative=False):       
        y_values = y.argmax(axis=1)
        m = y_values.shape[0]
        grad = x        
        # (Derivaitve unused as cross entropy only used seperately to calculate loss from softamx output)
        if derivative:
            grad[range(m), y_values] -= 1
            grad = grad / m
            return grad       
        log_likelihood = -np.log(grad[range(m), y_values])
        loss = np.sum(log_likelihood) / m    
        return loss

    
    
    # Choose correct activation function from our neural network
    def choose_activation(self, value, a_func, prop_backwards):
        # Choose the correct activation function from the designed neural network
        if a_func == "relu":
            correct_func = self.relu
        elif a_func == "sigmoid":
            correct_func = self.sigmoid
        elif a_func == "softmax":
            correct_func = self.softmax
        
        # Calls correct function and tells that function to use derivative if backward propagating
        if prop_backwards == False:
            return correct_func(value, derivative=False)
        elif prop_backwards == True:
            return correct_func(value, derivative=True)
    

    # Forward propagation of neural network
    def forward_propagation(self, x_train):
        params = self.params
        prop_backwards = False

        # input layer activations becomes input
        params['values0'] = x_train              
        
        # Moves from one layer to the next using the correct values from the parameters dictionary
        for index, layer in enumerate(self.NN):
            num = index + 1
            activation_fucntion = layer["activation"]
            params['Z' + str(num)] = np.dot(params['W' + str(num)], params['values' + str(num - 1)]) + params['B' + str(num)]
            params['values' + str(num)] = self.choose_activation(params['Z' + str(num)], activation_fucntion, prop_backwards)
        
        return params['values4']
    
    #Backward propagation of neural network
    def backward_propagation(self, y_train, output):
        params = self.params
        # Passed into activation functions telling them to use derivative
        prop_backwards = True
        # Temp value used to choose correct error calculation
        temp_val = True
              
        weight_updates = {}
        bias_updates = {}
        
        # Moves from one layer to the next in reverse using the correct values from the parameters dictionary
        for index, layer in reversed(list(enumerate(self.NN))):
            num = index + 1
            #print(num)
            activation_fucntion = layer["activation"]
            
            # Error calcultion if statement as clacualtion is differnt for first layer in back propagation
            if temp_val == True:
                error = 2 * (output - y_train) / output.shape[0] * self.choose_activation(params['Z' + str(num)], activation_fucntion, prop_backwards)            
                temp_val = False
            else:          
                error = np.dot(params['W' + str(num + 1)].T, error) * self.choose_activation(params['Z' + str(num)], activation_fucntion, prop_backwards)
                
            weight_updates['W' + str(num)] = np.outer(error, params['values' + str(num - 1)])
            # Calculate bias update
            bias_updates['B' + str(num)] = error * self.choose_activation(params['Z' + str(num)], activation_fucntion, prop_backwards)
        
        return weight_updates, bias_updates

    # Update fucntion for weights and biases using stored values for adjustment
    def update_network_parameters(self, changes_to_weight, changes_to_bias):
        # Updating of weights
        for keyW, valueW in changes_to_weight.items():
            self.params[keyW] -= self.l_rate * valueW
        
        # Updating of biases
        for keyB, valueB in changes_to_bias.items():
            self.params[keyB] -= self.l_rate * valueB
            
    # Function for computation of accuracy using validation data to check the neural networks progress
    def compute_accuracy(self, x_val, y_val):      
        predictions = []
        # Uses validation data to calaculate an accuracy percentage so we can see how well the training of our neural network is doing
        for x, y in zip(x_val, y_val):
            output = self.forward_propagation(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    # Training loop of the neural network where the fucntions are actually called
    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        
        # Arrays to store inforamtion used for plotting graphs
        accuracy_values = []
        loss_values = []
        
        # For loop for each epoch
        for iteration in range(self.epochs):
            outputs = []
            # For loop for each image passed into the neural network
            for x,y in zip(x_train, y_train):
                output = self.forward_propagation(x)
                outputs.append(output)
                changes_to_weight, changes_to_bias = self.backward_propagation(y, output)
                self.update_network_parameters(changes_to_weight, changes_to_bias)
                
            outputs = np.array(outputs)
            
            loss = self.cross_entropy(outputs, y_train)            
            accuracy = self.compute_accuracy(x_val, y_val)        
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%, Loss: {3: .4f} '.format(iteration+1, time.time() - start_time, accuracy * 100, loss))

            # Adds current accuracy and loss values to thier respective arrays
            accuracy_values.append(accuracy)          
            loss_values.append(loss)
            
        #print(accuracy_values)
        
        return accuracy_values, loss_values
       
dnn = DeepNeuralNetwork(neuralnetwork)
# Values allow you to train with only part of the data, this was used for testing purposes and can be removed
accuracy, loss = dnn.train(x_train[:60000], y_train[:60000], x_val[:10000], y_val[:10000])

print(accuracy)
print(loss)







