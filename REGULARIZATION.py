import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import math
from statistics import mean
style.use('fivethirtyeight')

#please change the dataset here
f = open('data1_new.pkl','rb')
#f = open('data2_new.pkl','rb')
#f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]

def best_fit_slope_and_intercept(X,Y):
    m = (np.multiply(mean(X),mean(Y)) -mean(np.multiply(X,Y) ))/((mean(X))**2-mean((X)**2))
    b = mean(Y) - m*mean(X)    
    return m,b
m,b = best_fit_slope_and_intercept(X,Y)
#print(m,b)
regression_line = [(m*x)+b for x in X]
plt.scatter(X,Y)
plt.plot(X,regression_line)


matrix_x = np.ones((X.shape[0],2))
matrix_x[:,0] = np.ones(X.shape[0])
matrix_x[:,1] = X
L = 5 
LI = [[0,0],[0,L]]    
model_parameters = np.matmul(np.matmul(np.linalg.inv(np.matmul(matrix_x.T,matrix_x) + LI),matrix_x.T),Y)
print(model_parameters)

plt.plot(X, model_parameters[0] + model_parameters[1]*X, color = 'black')
 

plt.title("line fitting with Regularization")
#plt.legend(loc = 2)
plt.axis([-150,150,-100,100])
plt.show()
