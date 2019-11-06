# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:31:03 2019

@author: Aditya's HP Omen 15
"""
# this program if for line fitting using OLS (oridnary least squares)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from statistics import mean
from numpy import linalg as LA

style.use('fivethirtyeight')

# DATA1
#importing data
f = open('data1_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.axis([-150,150,-100,100])

#creatng OLS function
def OLS(X,Y):
    m = (np.multiply(mean(X),mean(Y)) -mean(np.multiply(X,Y) ))/((mean(X))**2-mean((X)**2))
    c = mean(Y) - m*mean(X)    
    return m,c

m,c = OLS(X,Y)
regression_line = [(m*x)+c for x in X]
plt.figure(1)
plt.scatter(X,Y, color = 'blue')
#plotting line using OLS parameters
plt.plot(X,regression_line, color = 'red', label = "OLS")
X2 = np.array([da[:,0],X])
Y2 = np.array([da[:,1],regression_line])
plt.plot(X2,Y2,color = 'red')
plt.title("Line fitting with vertical least squares on data 1")
plt.legend()
plt.axis([-150,150,-100,100])
plt.show()

#DATA2
f = open('data2_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.axis([-150,150,-100,100])
#creatng OLS function
def OLS(X,Y):
    m = (np.multiply(mean(X),mean(Y)) -mean(np.multiply(X,Y) ))/((mean(X))**2-mean((X)**2))
    c = mean(Y) - m*mean(X)    
    return m,c
m,c = OLS(X,Y)
regression_line = [(m*x)+c for x in X]
plt.figure(2)
plt.scatter(X,Y, color = 'blue')
#plotting line using OLS parameters
plt.plot(X,regression_line, color = 'red', label = "OLS")
X2 = np.array([da[:,0],X])
Y2 = np.array([da[:,1],regression_line])
plt.plot(X2,Y2,color = 'red')
plt.title("Line fitting with vertical least squares on data 2")
plt.legend()
plt.axis([-150,150,-100,100])
plt.show()

#DATA3
f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.axis([-150,150,-100,100])
def OLS(X,Y):
    m = (np.multiply(mean(X),mean(Y)) -mean(np.multiply(X,Y) ))/((mean(X))**2-mean((X)**2))
    c = mean(Y) - m*mean(X)    
    return m,c
m,c = OLS(X,Y)
regression_line = [(m*x)+c for x in X]
plt.figure(3)
plt.scatter(X,Y, color = 'blue')
plt.plot(X,regression_line, color = 'red', label = "OLS")
X2 = np.array([da[:,0],X])
Y2 = np.array([da[:,1],regression_line])
plt.plot(X2,Y2,color = 'red')
plt.title("Line fitting with vertical least squares on data 3")
plt.legend()
plt.axis([-150,150,-100,100])
plt.show()
