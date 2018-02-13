import scipy.io
import numpy as np
import random
import time
which_power=int(input("to which power:    "))
which_method=int(input("0: GradDesc, 1: closeform    "))
mat = scipy.io.loadmat('/Users/yzh/Desktop/cour/supervised/HW1_Data/dataset1.mat')

def getPolyMat(input, power): #get poly feature matrix
  result=np.power(input,0)
  for i in range(1,power+1):
      result=np.concatenate((result,np.power(input,i)),axis=1)
  return result

train_X=getPolyMat(mat['X_trn'], which_power)
test_X=getPolyMat(mat['X_tst'], which_power)

def closeFormRegression(train, label):
    return np.matmul(np.matmul( np.linalg.inv( np.matmul(train.transpose(),
                                                         train) ),  train.transpose() ), label)

def GradDesc(train, label, learnRate, stopWhen, batchSize):# stochastic GD
    theta1=np.zeros((train.shape[1],1))
    multiply=learnRate*2;
    while(True):
      batch=random.sample(range(0,label.shape[0]),batchSize)
      theta2 = theta1 - multiply * np.matmul(train[batch,:].transpose(), np.matmul(train[batch,:],
                                    theta1)- label[batch,:])
      if np.linalg.norm(theta2-theta1)<=stopWhen:
          return theta2
      theta1=theta2

if which_method==0: # choose regression method
  per_batch_number = int(input("per-batch-number:  ")) #choose per-batch-number
  start=time.time()
  parameters=GradDesc(train_X, mat['Y_trn'], 0.00000005, 0.0001, per_batch_number)
  stop=time.time()
else:
    start = time.time()
    parameters = closeFormRegression(train_X, mat['Y_trn'])
    stop = time.time()

print("time in seconds: ", stop-start)
print("theta:  ",parameters)
print("train error by root-mean-square: ", np.linalg.norm( np.matmul(train_X,parameters)-mat['Y_trn'] )/mat['Y_trn'].shape[0]**0.5)#train error
print("test error by root-mean-square: ", np.linalg.norm( np.matmul(test_X,parameters)-mat['Y_tst'] )/mat['Y_tst'].shape[0]**0.5)#test error
