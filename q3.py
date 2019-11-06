#importing libraries
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import math
from statistics import mean
style.use('fivethirtyeight')

# =============================================================================
# #please change the dataset here by uncommenting the lines
# =============================================================================
f = open('data1_new.pkl','rb')
#f = open('data2_new.pkl','rb')
#f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]

#taking mean of the data
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
plt.scatter(X,Y, color = 'blue')
plt.plot(X, slope*X+intercept,color = 'red', label = 'line fit without RANSAC')
plt.axis([-150,150,-100,100])
plt.show()

#writing the function for line fitting
def line_fitting(X_and_Y_values):
    X = X_and_Y_values[:,0]
    Y = X_and_Y_values[:,1]
    m = (Y[1]-Y[0])/X[1]-X[0]
    c = Y[1] - m*X[1]
    return m,c

# calculating the distance from the line
def dist_from_line(points,slope,intercept):
    m = slope
    c = intercept
    distance=0
    distance = (abs(c+(m*points[0])-points[1]))/np.sqrt(1 + m**2)
    return distance

#initializing parameters
threshold_distance = 10
best_a = np.array([])
global_inliers = 0

#for loop for number of iterations
for count in range(500):
    local_inliers = 0
    random_points = np.random.choice(da.shape[0],2,replace = False)
    initial_two_points = da[random_points,:]
    m,c = line_fitting(initial_two_points)
    
    #calculating the distance of all the points from the line and sorting them
    for distance in da:
        d = dist_from_line(distance,m,c)
        if d<= threshold_distance:
            local_inliers += 1
    
    #updating the number of inliers
    if global_inliers <= local_inliers:
        global_inliers = local_inliers
        best_a = initial_two_points

#print("no. of inliers: "+ str(global_inliers))
best_fit_slope, best_fit_intercept = line_fitting(best_a)
y = (best_fit_slope*X + best_fit_intercept)
plt.plot(X,y,color = 'green', label = 'line fit with RANSAC')
plt.show()

