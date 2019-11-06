# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:54:42 2019

@author: Aditya's HP Omen 15
"""
#importing libraries
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import math
from statistics import mean
from numpy import linalg as LA

style.use('fivethirtyeight')

#-----------DATA1-----------------
#loading data 1
f = open('data1_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
#mean of X coordinates
X_bar = mean(X)
#mean of X coordinates
Y_bar = mean(Y)
delta_X = X - X_bar
delta_Y = Y - Y_bar
#calculating variance
variance_X = (np.sum(delta_X**2))/(X.shape[0]-1) 
variance_Y = (np.sum(delta_X**2))/(Y.shape[0]-1) 

delta_XY = np.multiply(delta_X,delta_Y)
#calculating covariance
covarience_XY = np.sum(delta_XY)/(X.shape[0]-1) 
covariance_matrix = np.array([[variance_X, covarience_XY], [covarience_XY, variance_Y]])
#calculating model parameters
slope = np.sum((delta_X)*(delta_Y))/np.sum(delta_X**2)
intercept = mean(Y)-slope*mean(X)
print("manually calculated covariance matrix:", covariance_matrix)
print("now we print covarience matric using np.cov to show that the matrix we have calculated is coorect")
cov_matrix = np.cov(da[:,0],da[:,1])
#calculating the eigen vectors and value from the covariance matrix
w, v = LA.eig(cov_matrix)
print("\n\n eigen values w: " ,w)
print("\n\n eigen vectors v: ",v)
origin = [mean(X)],[mean(Y)]
vector1 = w[0]*v[:,0]
vector2 = w[1]*v[:,1]
plt.figure(1)
plt.quiver(*origin, vector1[0], vector1[1], color=['r'], scale = 10000)
plt.quiver(*origin, vector2[0], vector2[1], color=['b'], scale = 10000)
plt.scatter(X,Y)
plt.plot(X, slope*X+intercept,color = 'green')
plt.title("Eigen vectors plotting for data set 1")
plt.axis([-150,150,-100,100])
plt.show()

#DATA2
f = open('data2_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]

X_bar = mean(X)
Y_bar = mean(Y)
delta_X = X - X_bar
delta_Y = Y - Y_bar

variance_X = (np.sum(delta_X**2))/(X.shape[0]-1) 
variance_Y = (np.sum(delta_X**2))/(Y.shape[0]-1) 

delta_XY = np.multiply(delta_X,delta_Y)
covarience_XY = np.sum(delta_XY)/(X.shape[0]-1) 
covariance_matrix = np.array([[variance_X, covarience_XY], [covarience_XY, variance_Y]])
slope = np.sum((delta_X)*(delta_Y))/np.sum(delta_X**2)
intercept = mean(Y)-slope*mean(X)
print("covariance matrix:", covariance_matrix)
cov_matrix = np.cov(da[:,0],da[:,1])
w, v = LA.eig(cov_matrix)
print("\n\nw: " ,w)
print("\n\nv: ",v)
origin = [mean(X)],[mean(Y)]
vector1 = w[0]*v[:,0]
vector2 = w[1]*v[:,1]
plt.figure(2)
plt.quiver(*origin, vector1[0], vector1[1], color=['r'], scale = 10000)
plt.quiver(*origin, vector2[0], vector2[1], color=['b'], scale = 10000)
plt.scatter(X,Y)
plt.plot(X, slope*X+intercept,color = 'green')
plt.title("Eigen vectors plotting for data set 2")
plt.axis([-150,150,-100,100])
plt.show()


#DATA3
f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]

X_bar = mean(X)
Y_bar = mean(Y)
delta_X = X - X_bar
delta_Y = Y - Y_bar

variance_X = (np.sum(delta_X**2))/(X.shape[0]-1) 
variance_Y = (np.sum(delta_X**2))/(Y.shape[0]-1) 

delta_XY = np.multiply(delta_X,delta_Y)
covarience_XY = np.sum(delta_XY)/(X.shape[0]-1) 
covariance_matrix = np.array([[variance_X, covarience_XY], [covarience_XY, variance_Y]])
slope = np.sum((delta_X)*(delta_Y))/np.sum(delta_X**2)
intercept = mean(Y)-slope*mean(X)
print("covariance matrix:", covariance_matrix)
cov_matrix = np.cov(da[:,0],da[:,1])
w, v = LA.eig(cov_matrix)
print("\n\nw: " ,w)
print("\n\nv: ",v)
origin = [mean(X)],[mean(Y)]
vector1 = w[0]*v[:,0]
vector2 = w[1]*v[:,1]
plt.figure(3)
plt.quiver(*origin, vector1[0], vector1[1], color=['r'], scale = 10000)
plt.quiver(*origin, vector2[0], vector2[1], color=['b'], scale = 10000)
plt.scatter(X,Y)
plt.plot(X, slope*X+intercept,color = 'green')
plt.title("Eigen vectors plotting for data set 3")
plt.axis([-150,150,-100,100])
plt.show()
