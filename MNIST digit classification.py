import numpy as np
import sklearn
from sklearn.metrics import *
import matplotlib.pyplot as plt
import random

def training(training_size=60000):
    subplot = 220
    for hu in hidden_units:
        for m in momentum:
            test_accuracies = []
            train_accuracies = []
            epochs = []
            #initialze weights for input to hidden layer
            weights_input_hidden = np.random.uniform(-0.05, 0.05, [hu, input])
            #initialze weights for hidden to output layer
            weights_hidden_output = np.random.uniform(-0.05, 0.05, [output, hu + 1])
          
            previous_weights_hidden_output = np.zeros(weights_hidden_output.shape)
            previous_weights_input_hidden = np.zeros(weights_input_hidden.shape)

            for epoch in range(num_epochs + 1):
                print("End of Epoch", epoch)
                if epoch == 0:
                    # Accuracies for test and train data for epoch 0
                    correctly_classified = 0
                    train_accuracies = calculate_training_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, train_accuracies, training_size)
                    correctly_classified = 0
                    test_accuracies = calculate_testing_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, test_accuracies)
                else:
                    for count, train_data in enumerate(training_data):
                        if count <= training_size:
                            # Accessing the test data
                            data = (train_data.strip()).split(",")
                            # First column in a test example is the actual target
                            target = int(data[0])
                            data = data[1:]
                            # Preprocessing
                            input_data = ((np.asfarray(data)) / 255)
                            targets = []

                            for i in range(output):
                                if i == target:
                                    targets.append(0.9)
                                else:
                                    targets.append(0.1)

                            # Training the data
                            previous_weights_hidden_output, previous_weights_input_hidden, weights_hidden_output, weights_input_hidden = train(input_data, weights_input_hidden, hu, weights_hidden_output, targets, m, previous_weights_hidden_output, previous_weights_input_hidden)
                    #calculate the train and test accuracy
                    correctly_classified = 0
                    train_accuracies = calculate_training_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, train_accuracies, training_size)
                    correctly_classified = 0
                    test_accuracies = calculate_testing_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, test_accuracies)

                epochs.append(epoch)
                
			#plot the graph after training and testing	
            subplot += 1
            plt.figure(1, figsize=(10, 8))
            plt.subplot(subplot)
            plt.title("Hidden Units: %s, Momentum: %s" % (hu, m))
            plt.plot(epochs, test_accuracies, label='Test Data')
            plt.plot(epochs, train_accuracies, label='Train Data')
            plt.legend(loc='lower right')
            plt.ylabel("Accuracy")
            plt.yticks(range(0, 100, 10))
            plt.ylim(0, 100)
            plt.xlabel("Epoch")
            plt.tight_layout()
    plt.show()

def calculate_training_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, train_accuracies, training_size):
    for count, train_data in enumerate(training_data):
        if count <= training_size:
            data = (train_data.strip()).split(",")
            target = int(data[0])
            data = data[1:]
            input_data = ((np.asfarray(data)) / 255)
            input_data = (np.append([1], input_data)).T
            #dot product of weigths and input data
            hidden_input = np.dot(weights_input_hidden, input_data)
            #sigmoid activation function for input to hidden
            hidden_activation = 1 / (1 + np.exp(-hidden_input))
            hidden_activation = (np.append([1], hidden_activation)).T 
            #dot product of weights and hidden node activation
            output_input = np.dot(weights_hidden_output, hidden_activation)
            #sigmoid activation function for hidden to output
            output_activation = 1 / (1 + np.exp(-output_input))
            highest_output = np.argmax(output_activation)
            #calculate the correctly classified labels
            if highest_output == target:
                correctly_classified += 1
    accuracy = (float(correctly_classified) / float(training_size)) * 100
    print("Accuracy of training data:", accuracy)
    train_accuracies.append(accuracy)
    return train_accuracies

def calculate_testing_accuracy(correctly_classified, weights_input_hidden, weights_hidden_output, test_accuracies):
    for test_data in testing_data:
        data = (test_data.strip()).split(",")
        target = int(data[0])
        data = data[1:]
        input_data = ((np.asfarray(data)) / 255)
        input_data = (np.append([1], input_data)).T
        #dot product of weigths and input data
        hidden_input = np.dot(weights_input_hidden, input_data)
        #sigmoid activation function for input to hidden
        hidden_activation = 1 / (1 + np.exp(-hidden_input))
        hidden_activation = (np.append([1], hidden_activation)).T
        #dot product of weights and hidden node activation
        output_input = np.dot(weights_hidden_output, hidden_activation)
        #sigmoid activation function for hidden to output
        output_activation = 1 / (1 + np.exp(-output_input))
        highest_output = np.argmax(output_activation)
        #calculate the correctly classified labels
        if highest_output == target:
            correctly_classified += 1
    accuracy = (float(correctly_classified) / float(len(testing_data))) * 100
    print("Accuracy of testing data:", accuracy)
    test_accuracies.append(accuracy)
    return test_accuracies

def compute_confusion_matrix(weights_input_hidden, weights_hidden_output, prediction_list, target_list):
    for test_data in testing_data:
        data = (test_data.strip()).split(",")
        target = int(data[0])
        data = data[1:]
        input_data = ((np.asfarray(data)) / 255)
        input_data = (np.append([1], input_data)).T
        hidden_input = np.dot(weights_input_hidden, input_data)
        hidden_activation = 1 / (1 + np.exp(-hidden_input))
        hidden_activation = (np.append([1], hidden_activation)).T
        output_input = np.dot(weights_hidden_output, hidden_activation)
        output_activation = 1 / (1 + np.exp(-output_input))
        highest_output = np.argmax(output_activation)
        prediction_list.append(highest_output)
        target_list.append(target)
    print("\n Confusion Matrix on the test set:")
    print(confusion_matrix(target_list, prediction_list))

def train(input_data, weights_input_hidden, hu, weights_hidden_output, targets, momentum, previous_weights_hidden_output, previous_weights_input_hidden):
    input_data = (np.append([1], input_data)).T
    hidden_input = np.dot(weights_input_hidden, input_data)
    #sigmoid activation of input to hidden layer
    hidden_activation = 1 / (1 + np.exp(-hidden_input))
    hidden_activation = (np.append([1], hidden_activation)).T
    output_input = np.dot(weights_hidden_output, hidden_activation)
    #sigmoid activation of hidden to output layer
    output_activation = 1 / (1 + np.exp(-output_input))
    target_array = np.asfarray(targets)
    #calculate error at output and error at activation
    output_error = output_activation * (1 - output_activation) * (target_array - output_activation)
    hidden_unit_error = hidden_activation * (1 - hidden_activation) * np.dot(output_error, weights_hidden_output)
    hidden_activation = np.array(hidden_activation, ndmin=2)
    output_error = np.array(output_error, ndmin=2)
    #update the weights for output to hidden
    delta_weights_hidden_output = (learning_rate * np.dot(output_error.T, hidden_activation)) + (momentum * previous_weights_hidden_output)
    hidden_unit_error = np.array(hidden_unit_error, ndmin=2)
    input_data = np.array(input_data, ndmin=2)
    #update the weights for hidden to input
    delta_weights_input_hidden = (learning_rate * np.dot(hidden_unit_error[:, 1:].T, input_data)) + (momentum * previous_weights_input_hidden)
    weights_hidden_output += delta_weights_hidden_output
    weights_input_hidden += delta_weights_input_hidden
    previous_weights_hidden_output = delta_weights_hidden_output
    previous_weights_input_hidden = delta_weights_input_hidden
    return previous_weights_hidden_output, previous_weights_input_hidden, weights_hidden_output, weights_input_hidden

# Reading the train dataset
train_file = open("/content/mnist_train.csv", "r")
training_data = train_file.readlines()
train_file.close()

# Reading the test dataset
test_file = open("/content/mnist_test.csv", "r")
testing_data = test_file.readlines()
test_file.close()

# Initializing input, outputs, learning rate, and epochs
input = 785
output = 10
num_epochs = 50
learning_rate = 0.1

# Experiment 1 
print("Experiment 1 - Vary hiddden units")
hidden_units = [20,50,100]
momentum = [0.9]
training()

# Experiment 2 
print("Experiment 2 - Vary momentum")
hidden_units = [100]
momentum = [0.0,0.25,0.5]
training()

# Experiment 3 
print("Experiment 3 - Vary training samples")
hidden_units = [100]
momentum = [0.9]
training(15000) 
training(30000)
