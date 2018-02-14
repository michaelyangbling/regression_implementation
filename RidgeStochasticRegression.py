import scipy.io
import numpy as np
import random
import time
from sklearn.cross_validation import KFold  # sklearn This module will be removed in 0.20.", DeprecationWarning
print("please first input some commands")
print("   ")
which_power=int(input("poly to which power:    "))
foldNum=int(input("how many folds for k-fold CV:    "))
which_method=int(input("0: GradDesc, 1: closeform    "))
mat = scipy.io.loadmat('/Users/yzh/Desktop/cour/supervised/HW1_Data/dataset1.mat')

def getPolyMat(input, power): #get poly feature matrix
  result=np.power(input,0)
  for i in range(1,power+1):
      result=np.concatenate((result,np.power(input,i)),axis=1)
  return result

train_X=getPolyMat(mat['X_trn'], which_power)
test_X=getPolyMat(mat['X_tst'], which_power)
Y_trn=mat['Y_trn']

def get_error(train, theta, test): # get rmse( root mean square error)
    return np.linalg.norm( np.matmul(train,theta)-test )/test.shape[0]**0.5

def RidgecloseForm(train, label,avoidOverFit):# β=[(XtX+λI)^−1]XtY
    return np.matmul(np.matmul( np.linalg.inv( np.matmul(train.transpose(),
             train) + avoidOverFit*np.identity(train.shape[1]) ), train.transpose() ), label)

def RidgeGradDesc(train, label, learnRate, stopWhen, batchSize,avoidOverFit):# stochastic GD
    theta1=np.zeros((train.shape[1],1))
    multiply=learnRate*2;
    while(True):
      batch=random.sample(range(0,label.shape[0]),batchSize)
      theta2 = theta1 - multiply * (np.matmul(train[batch,:].transpose(), np.matmul(train[batch,:],
                                    theta1)- label[batch,:])+avoidOverFit*theta1)
      if np.linalg.norm(theta2-theta1)<=stopWhen:
          return theta2
      theta1=theta2

if which_method==0: # choose regression method
  per_batch_number = int(input("per-batch-number:  ")) #choose per-batch-number
  start=time.time()

  kf = KFold(train_X.shape[0], n_folds=foldNum)
  pararid = (100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0) #k-fold cv
  paraErrorDict = {}
  for avoidOverFit in pararid:
      sum = 0;
      count = 0;
      for train_index, test_index in kf:
          Xtrain, Xtest = train_X[train_index], train_X[test_index]
          ytrain, ytest = Y_trn[train_index], Y_trn[test_index]
          sum += get_error(Xtest, RidgeGradDesc(Xtrain, ytrain, 0.00000005, 0.0001, per_batch_number, avoidOverFit), ytest);
          count += 1
      paraErrorDict[avoidOverFit] = sum / count

  optAvoidOverFit = pararid[0]  # get best hyper-parameter
  for i in range(1, len(pararid)):
      if paraErrorDict[pararid[i]] < paraErrorDict[optAvoidOverFit]:
          optAvoidOverFit = pararid[i]
  parameters=RidgeGradDesc(train_X, Y_trn, 0.00000005, 0.0001, per_batch_number, optAvoidOverFit)
  stop=time.time()
else: # close form version

    start = time.time()

    kf = KFold(train_X.shape[0], n_folds=foldNum)
    pararid = (100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0)
    paraErrorDict = {}
    for avoidOverFit in pararid:
        sum = 0;
        count = 0;
        for train_index, test_index in kf:
            Xtrain, Xtest = train_X[train_index], train_X[test_index]
            ytrain, ytest = Y_trn[train_index], Y_trn[test_index]
            sum += get_error(Xtest, RidgecloseForm(Xtrain, ytrain, avoidOverFit), ytest);
            count += 1
        paraErrorDict[avoidOverFit] = sum / count

    optAvoidOverFit = pararid[0]
    for i in range(1, len(pararid)):
        if paraErrorDict[pararid[i]] < paraErrorDict[optAvoidOverFit]:
            optAvoidOverFit = pararid[i]
    parameters = RidgecloseForm(train_X, Y_trn, optAvoidOverFit)
    stop = time.time()

print("follwing are outputs:")
print("   ")
print("best hyper-parameter:  ", optAvoidOverFit)
print("theta:  ",parameters)
print("train error by root-mean-square:    ", get_error(train_X, parameters, Y_trn))#train error
print("test error by root-mean-square:    ", get_error(test_X, parameters, mat['Y_tst']))#test error
print("How does the test error change as a function of hyper-parameter")
for j in range(0,len(pararid)):  #print test errors for all hyperparameters
    if which_method==0:
      p=RidgeGradDesc(train_X, Y_trn, 0.00000005, 0.0001, per_batch_number, pararid[j])
      print(pararid[j], "   ",get_error(test_X, p, mat['Y_tst'] ))
    else:
      p = RidgecloseForm(train_X, Y_trn, pararid[j])
      print(pararid[j], "   ",get_error(test_X, p, mat['Y_tst']))

