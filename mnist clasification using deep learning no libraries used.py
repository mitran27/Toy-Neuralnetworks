# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:11:23 2020

@author: Mitran
"""




# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:24:22 2020

@author: Mitran
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:15:40 2020

@author: Mitran
"""
import numpy as np
import pymongo
import pickle 

import gzip



client='your db url'
db='your db'
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

def softmax(raw_preds):
    
  
    return  np.exp(raw_preds)/np.sum( np.exp(raw_preds))
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
def categoricalCrossEntropy(label,probs):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs)) # Multiply the desired output label by the log of the prediction, then sum all values in the vector


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
class Flatten(layer):
    # returns the flattened input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.reshape((1,-1))
        #print(self.output.shape)
       
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):# not needed simply ofr code
        return output_error.reshape(self.input.shape)      

class Dense(layer):#fully connected layer only two layers of a nn
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

class Conv2D(layer):
    def __init__(self,size,stride):
        scale=1.0
        stddev = scale/np.sqrt(np.prod(size))
        self.kernel= np.random.normal(loc = 0, scale = stddev, size = size)
        #self.kernel=np.array([[[[1,2,4],[0,1,1],[3,2,1]],[[1,1,1],[1,1,1],[1,1,1]]],[[[3,2,0],[0,1,1],[3,2,1]],[[1,0,0],[0,0,0],[0,0,0]]]])
        self.bias=np.ones((self.kernel.shape[0],1))
        self.stride=stride
       
        self.der_kernel=np.zeros((self.kernel.shape))
        self.momentum_kernel=np.zeros((self.kernel.shape))
        self.adam_kernel=np.zeros((self.kernel.shape))
        self.der_bias=np.zeros((self.bias.shape))
        self.momentum_bias=np.zeros((self.bias.shape))
        self.adam_bias=np.zeros((self.bias.shape))
        
       
       
        
    def forward_propagation(self,inp_data):     
        #initially we need a kernel of size()
      self.input=inp_data
     
      image=inp_data
      out=self.kernel.shape
      
      
      no_of_filters=out[0]
      n_c_f=out[1]
      filter_dim=out[2] # filter dimensions
     
      n_c, img_dim, _ = image.shape # image dimensions
      
      feature_map_dim = int((img_dim-filter_dim)/self.stride)+1 # calculate output dimensions
      f=filter_dim
      s=self.stride
      feature_map_matrix=np.zeros((no_of_filters,feature_map_dim,feature_map_dim))
      fmm=feature_map_matrix
    
    # convolve each filter over the image
      for filter_no,cf in enumerate(self.kernel):#each filter
        
          inp_img_y=inp_img_x=fm_pos_y=fm_pos_x=0
          while (inp_img_y+filter_dim<=img_dim):
            
            # fir one row finished at next row column is made zero 
              inp_img_x=fm_pos_x=0
            #img will take uptosize of kernel in inputimg but in feature map only in idex
              while(inp_img_x+filter_dim<=img_dim):
               
                #the convolution........vvvvv important
                   
                   img_slice=image[:,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f]
                  
                   fmm[filter_no, fm_pos_y, fm_pos_x]=np.sum(cf*img_slice)+self.bias[filter_no]
                   inp_img_x+=s
                   fm_pos_x+=1
              inp_img_y+=s
              fm_pos_y+=1
            
      return  fmm










    def update_parameters(self,learning_rate,der_kernel,der_bias):
        """old_kernel=self.kernel
        old_bias=self.bias
        self.kernel=old_kernel-learning_rate*der_kernel"""
        self.der_kernel+=der_kernel
        self.der_bias+=der_bias
        batch_size=20
        beta1=0.95
        beta2=0.99
        self.momentum_kernel = beta1*  self.momentum_kernel + (1-beta1)*self.der_kernel/batch_size # momentum update
        self.adam_kernel = beta2*self.adam_kernel + (1-beta2)*(self.der_kernel/batch_size)**2 # RMSProp update
        self.kernel -= learning_rate * self.momentum_kernel/np.sqrt(self.adam_kernel+1e-7) # combine momentum and RMSProp to perform update with Ad
        
        
        self.momentum_bias = beta1*  self.momentum_bias + (1-beta1)*self.der_bias/batch_size # momentum update
        self.adam_bias = beta2*self.adam_bias + (1-beta2)*(self.der_bias/batch_size)**2 # RMSProp update
        self.bias -= learning_rate * self.momentum_bias/np.sqrt(self.adam_bias+1e-7) # combine momentum and RMSProp to perform update with Ad
       
    def backward_propagation(self, output_error, learning_rate):
        
        f=self.kernel.shape[2]
       
        s=self.stride
        n_c, img_dim, _ = self.input.shape # image dimensions
        der_next_layer=np.zeros((self.input.shape))
        (no_of_filters, n_c_f, filter_dim, _) = self.kernel.shape # filter dimensions
        der_filt = np.zeros(self.kernel.shape)
        der_bias = np.zeros((self.kernel.shape[0],1))
        for filter_no,cf in enumerate(self.kernel):#each filter
        
          inp_img_y=inp_img_x=fm_pos_y=fm_pos_x=0
          while (inp_img_y+filter_dim<=img_dim):
            
          
              inp_img_x=fm_pos_x=0
            
              while(inp_img_x+filter_dim<=img_dim):
               
                #the convolution. backward.......vvvvv important
                  
                  # we are using summation i form of +=
                   
                        
                   der_filt[filter_no] += output_error[filter_no, fm_pos_y, fm_pos_x] * self.input[:,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f]
                 
                   y=(output_error[filter_no,fm_pos_y,fm_pos_x]*cf)
                  
                   der_next_layer[:,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f]+=y
                  
                  
                  
                  
                   inp_img_x+=s
                   fm_pos_x+=1
              inp_img_y+=s
              fm_pos_y+=1
          der_bias[filter_no] = np.sum(output_error[filter_no])
        self.update_parameters(learning_rate,der_filt,der_bias)
            
        return  der_next_layer
    
    
    
class MaxPooling2D(layer):
    def __init__(self,filter_size,stride):
        self.f_size=filter_size
        self.stride=stride
        
    def forward_propagation(self,inp_img):
       self.input=inp_img
       f=self.f_size
       s=self.stride
       img=inp_img
       n_c, inp_dim_y, inp_dim_x = inp_img.shape
    
       # calculate output dimensions after the maxpooling operation.
       x = int((inp_dim_y - f)/s)+1 
       y = int((inp_dim_x - f)/s)+1
       pool_layer = np.zeros((n_c, x, y)) 
       for i in range(img.shape[0]):#each color
          
           inp_img_y=inp_img_x=fm_pos_y=fm_pos_x=0
           while (inp_img_y+f<=inp_dim_y):
            
            # fir one row finished at next row column is made zero 
               inp_img_x=fm_pos_x=0
            #img will take uptosize of kernel in inputimg but in feature map only in idex
               while(inp_img_x+f<=inp_dim_x):
                
                #the convolution........vvvvv important
                
                    pool_layer[i, fm_pos_y, fm_pos_x]=np.max(img[i,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f])
                    inp_img_x+=s
                    fm_pos_x+=1
               inp_img_y+=s
               fm_pos_y+=1
            
       return  pool_layer
  
    def backward_propagation(self, output_error, learning_rate):
       
       f=self.f_size
       s=self.stride
       img=self.input
       n_c, inp_dim_y, inp_dim_x = img.shape
    
       # calculate output dimensions after the maxpooling operation.
       x = int((inp_dim_y - f)/s)+1 
       y = int((inp_dim_x - f)/s)+1
       der_next_layer = np.zeros((self.input.shape)) 
       for i in range(img.shape[0]):#each color
          
           inp_img_y=inp_img_x=fm_pos_y=fm_pos_x=0
           while (inp_img_y+f<=inp_dim_y):
            
            # fir one row finished at next row column is made zero 
               inp_img_x=fm_pos_x=0
            #img will take uptosize of kernel in inputimg but in feature map only in idex
               while(inp_img_x+f<=inp_dim_x):
                
                #the convolution........vvvvv important
                    arr=np.array(img[i,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f])
                    idx = np.nanargmax(arr)
                    a, b = np.unravel_index(idx, arr.shape)
                  
                    der_next_layer[i,inp_img_y+a,inp_img_x+b]=output_error[i,fm_pos_y,fm_pos_x]
                    #der_prev_layerpool_layer[i, fm_pos_y, fm_pos_x]=np.max(img[i,inp_img_y:inp_img_y+f,inp_img_x:inp_img_x+f])
                    inp_img_x+=s
                    fm_pos_x+=1
               inp_img_y+=s
               fm_pos_y+=1
     
       return  der_next_layer   
def findthval(a,b):
    a=a.reshape(10,1)
    n=a.T
    x=a.tolist()
    y=b[:][0]
    arr1 = np.array(y)
    val=max(x)
    arr =y.tolist()
    val1=max(arr)
    print(x.index(val),arr.index(val1))
   
class Activation(layer):
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
        err=1
         # we have to forwardpass+backpropagate=1 epoch
         #one epoch ->is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
        for i in range (0,epochs):
            print(epochs-i)           
           
            err=0
            for j in range(no_sample):
               
                layer_output=traindata[j].T
                for col,each_layer in enumerate(self.layers):
                 
                    layer_output=each_layer.forward_propagation(layer_output)
                # after crossing all the layers error is found for all inputs loss/error is found
                err = self.loss(trainlabel[j], layer_output)
                newop=softmax(layer_output)
                findthval(trainlabel[j], layer_output)
               
               
                #after error/loss is found backprop is done to adjust the parameters for optimising the solution
                # backtracking
                
                error = self.loss_prime(trainlabel[j], newop)
               
                for each_layer in reversed(self.layers):
                    error=each_layer.backward_propagation(error,learning_rate)
          
               
                
                                            
        
    def predict(self,testdata):
        print('predict')
        no_sample=len(testdata)
        result=[]
       
        for i in range(0, no_sample):            
           
                layer_output=testdata[i].T
                for each_layer in self.layers:
                    layer_output=each_layer.forward_propagation(layer_output)
                result.append(layer_output,)
        return result



from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(x_train.shape)
x_train=x_train.reshape((60000,28,28,1))
model = network()




model.add(Conv2D((8,1,5,5),1))

model.add(Activation('tanh'))

model.add(Conv2D((8,8,5,5),1))
model.add(Activation('tanh'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(800,64))
model.add(Activation('tanh'))

model.add(Dense(64,10))
model.add(Activation('tanh'))
"""
afn='tanh'
model.add(Flatten())
model.add(Dense(5408, 100))               
model.add(Activation(afn))
model.add(Dense(100, 50))                   
model.add(Activation(afn))
model.add(Dense(50, 10))                    
model.add(Activation(afn))
"""
model.use(cal_err,cald_err_der)
model.use(cal_err, cald_err_der)
model.fit(x_train[0:500], y_train[0:500], epochs=10, learning_rate=0.1)

print(x_test.shape)
x_test=x_test.reshape((10000,28,28,1))
out = model.predict(x_test[0:20])


print("predicted values : ")
for k in out:
 for i in k:
    
    lsit=[]
    for j in i:
        lsit.append(j)
    val=max(lsit)
    print(lsit.index(val))
    
        
print("true values : ")
for i in (y_test[0:20]):
   
        for a,j in enumerate(i):# onlu one element will have one
            if(j==1):
                print(a)
                
                
i=save_model_mongo(model,'digit_classification')
        

# when using conv layers 19 was correct in 20  but with only dense layer 9 were correct


