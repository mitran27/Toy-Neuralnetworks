# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:15:40 2020

@author: Mitran
"""
import numpy as np
import pymongo
import pickle 

import gzip

#rmit2$ran%

client='mongodb+srv://edukit:mitran20@cluster0-m7fv6.mongodb.net/'
db='edukit'
dbcollection='network'

def save_model_mongo(model,modelname):

   net_mode=pickle.dumps(model)
   myclient=pymongo.MongoClient(client)
   mydb=myclient[db]
   mycon=mydb[dbcollection]
   info=mycon.insert_one({'model':net_mode,'name':modelname})
   return info
  
def load_model_mongo(modelname):
   net_mode={}
   myclient=pymongo.MongoClient(client)
   mydb=myclient[db]
   mycon=mydb[dbcollection]
   info=mycon.find({'name':modelname})
  
   for i in info:
       net_mode=i    
   model=pickle.loads(net_mode['model'])
   return model
def extract_data(filename, num_img, img_dim):
    
  
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(img_dim * img_dim * num_img)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_img, img_dim*img_dim)
        return data

def extract_labels(filename, num_img):
    
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_img)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels    

def tanh(x):
   
    return np.tanh(x);

def tanh_prime(x,o_r):
     
    
     y= 1-np.tanh(x)**2;
     return y*o_r
def sigmoid(x):       
   
     return 1/(1 + np.exp(-x))

def sigmoid_prime(x,o_r):    
     
        y=sigmoid(x)*(1-sigmoid(x))
        return y*o_r
   
# relu alone not working
def relu(x):    

    return  x*(x>0)
def relu_prime(x,o_r):
     
 

   o_r = np.array(o_r, copy = True)
   o_r[x <= 0] = 0.01;
   return o_r;



def cal_err(target, actual):
  
    return np.mean(np.power(target-actual, 2));


def cald_err_der(target, actual):
    return 2*(actual-target)/target.size;

class layer:
    def __init__(self):
        self.input=None
        self.output=None
    def forwardpass(inputs,weight):
        raise NotImplementedError
    def backpass(output,errorrate,learnrate):
        raise NotImplementedError
"""class dropoutlayer(layer):
    def __init__(self,prob_val):
        self.prob=prob_val
     def forward_propagation(self, input_data):
     def backward_propagation(self, output_error, learning_rate):
  """  
class FlattenLayer(layer):
    # returns the flattened input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.reshape((1,-1))
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):# not needed simply ofr code
        return 1      

class Fclayer(layer):#fully connected layer only two layers of a nn
    def __init__(self,inp_size,out_size):
        self.weights=np.random.rand(inp_size, out_size) - 0.5
        self.bias=np.random.rand(1, out_size) - 0.5
    def update_parameters(self,lr,weight,bias):
        self.weights-=lr*weight
        self.bias-=lr*bias
    def forward_propagation(self,inp_data):
        self.input = inp_data
      
        self.output =np.dot(inp_data,self.weights ) + self.bias
        return self.output
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        #print(weights_error.shape,self.input.T.shape,output_error.shape)
        # update parameters
        self.update_parameters(learning_rate,weights_error,output_error)
        """self.weights -= (learning_rate * weights_error)
        self.bias -= learning_rate * output_error"""
        return input_error
    

      


class ActivationLayer(layer):
    def __init__(self, activation):
        if(activation=='tanh'):
            self.activation = tanh
            self.activation_prime = tanh_prime
        if(activation=='relu'):
            self.activation = relu
            self.activation_prime = relu_prime
        if(activation=='sigmoid'):
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
       

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        output = self.activation(self.input)
        return output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
 
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input,output_error) 
class network:
    def __init__(self):
        self.layers=[]
        self.loss=None
        self.loss_prime=None
    def add(self,layer):
        self.layers.append(layer)
    def use(self,loss,loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime
    def fit(self,traindata,trainlabel,epochs,learning_rate):
        print('fit')
        no_sample=len(traindata)
         # we have to forwardpass+backpropagate=1 epoch
         #one epoch ->is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
        for i in range (0,epochs):
            print(epochs-i)
            err=0
            for j in range(no_sample):
                layer_output=traindata[j]
                for each_layer in self.layers:
                    layer_output=each_layer.forward_propagation(layer_output)
                # after crossing all the layers error is found for all inputs loss/error is found
                err += self.loss(trainlabel[j], layer_output)
              
                #after error/loss is found backprop is done to adjust the parameters for optimising the solution
                # backtracking
                error = self.loss_prime(trainlabel[j], layer_output)
                for each_layer in reversed(self.layers):
                    error=each_layer.backward_propagation(error,learning_rate)
               
               
                
                                            
        
    def predict(self,testdata):
        print('predict')
        no_sample=len(testdata)
        result=[]
        for i in range(0, no_sample):            
           
                layer_output=testdata[i]
                for each_layer in self.layers:
                    layer_output=each_layer.forward_propagation(layer_output)
                result.append(layer_output,)
        return result

checktn =0
checktnp =0
"""
activ="tanh"
net = network()
net.add(Fclayer(2, 3))
net.add(ActivationLayer(activ ))
net.add(Fclayer(3, 3))
net.add(ActivationLayer(activ))
net.add(Fclayer(3, 1))
net.add(ActivationLayer(activ))
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[0], [1], [1], [0]])
# train
net.use(cal_err, cald_err_der)

#Smaller learning rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs. ... If you have time to tune only one hyperparameter, tune the learning rate

net.fit(x_train, y_train, 10000, learning_rate=0.01)


out = net.predict(x_train)
for k in out:
 for i in k:
    
   
    for j in i:
        print(np.power(round(j,0),2))
""" 

activ="tanh"
from keras.datasets import mnist
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)


#x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)
afn='sigmoid'

net = network()
net.add(FlattenLayer())
net.add(Fclayer(28*28, 100))               
net.add(ActivationLayer(afn))
net.add(Fclayer(100, 50))                   
net.add(ActivationLayer(afn))
net.add(Fclayer(50, 10))                    
net.add(ActivationLayer(afn))


net.use(cal_err, cald_err_der)
net.fit(x_train[0:5000], y_train[0:5000], epochs=140, learning_rate=0.025)


out = net.predict(x_test[0:10])


print("predicted values : ")
for k in out:
 for i in k:
    
    lsit=[]
    for j in i:
        lsit.append(j)
    val=max(lsit)
    print(lsit.index(val))
    
        
print("true values : ")
for i in (y_test[0:10]):
   
        for a,j in enumerate(i):# onlu one element will have one
            if(j==1):
                print(a)


i=save_model_mongo(net,'digit_classification')





out = load_model_mongo('digit_classification').predict(x_test[0:10])

