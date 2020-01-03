# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=5, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs +1 )
           
    def predict(self, inputs):
       # print(np.dot(inputs, self.weights[1:]))
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        #print(summation)
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        sayac=0
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                print(self.weights[1:])
                self.weights[0] += self.learning_rate * (label - prediction)
                print("weight[0]:: ",self.weights[0])
                sayac+=1
                print("weights[0,0,0]= ",self.weights)
                print("**********")
            print(self.weights)
            print("+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("sayac :",sayac)

perceptron = Perceptron(2)
print(perceptron.weights)
print("-------------------------------------------------")
inputs = np.array([1, 1])
perceptron.predict(inputs)
print(perceptron.predict(inputs))

print("-----------------------------------------------")
training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

#perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

print("--------------------------------------------")
inputs = np.array([1, 1])
perceptron.predict(inputs) 
print(perceptron.predict(inputs))
print("--------------------------------------------")
#=> 1
inputs = np.array([0, 1])
perceptron.predict(inputs)  
print(perceptron.predict(inputs))
print(perceptron.weights)



 