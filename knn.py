#Kmeans Clustering from scratch with ANIMATION
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

#cluster size
c_num = 3

#specify proper path
df = pd.read_csv('knn.csv')
X = df.values

#sample size
s_num = X.shape[0]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim([ np.min(X[:,0])-5, np.max(X[:,0])+5])
ax.set_ylim([ np.min(X[:,1])-5, np.max(X[:,1])+5])
ax.set_title("Kmeans Running ....")
ax.scatter(X[:,0],X[:,1],c= 'C1',s=12, alpha = 0.2)

culster_centroids = []
culster_centroids_plot = []
iter_num = 1
#intial centroid plotting
#random points are selected as initial centroids
for i in range(c_num):
    ax.set_title("Kmeans Initial Centroids Assignment ....")
    r_num = np.random.randint(s_num)
    culster_centroids.insert(i,X[r_num])
    plt_pnt = ax.scatter(X[r_num,0],X[r_num,1],s = 500, c= 'C' + str(i+2),marker='+',edgecolors = 'face',alpha  =1)
    culster_centroids_plot.insert(i,plt_pnt )
    time.sleep(.2)
    fig.canvas.draw()
    fig.canvas.flush_events()
    

while True:
    ax.set_title("Kmeans Running Iteration "+ str(iter_num) + " ....")
    #assignment of datapoints to clusters
    datapoint_cluster_num = np.zeros(s_num)
    for i in range(s_num):
        dist_array = np.zeros([c_num]) #.reshape(-1,1)
        for j in range(len(culster_centroids)):
            dist_array[j] = np.sum(np.square(culster_centroids[j] - X[i]))
        
        color_index = np.argmin(dist_array)
        datapoint_cluster_num[i] = color_index
        ax.scatter(X[i,0],X[i,1],s = 12,c= 'C' + str(color_index+2),alpha = 0.2)
        time.sleep(0.02)
        
    flag_cnt = 0
    #centorids reassignment
    for i in range(c_num):
        length_cluster = len(X[datapoint_cluster_num == i][:,0])
        c_x_co_ordinate = np.sum(X[datapoint_cluster_num == i][:,0]) / length_cluster
        c_y_co_ordinate = np.sum(X[datapoint_cluster_num == i][:,1]) / length_cluster
        culster_centroids_plot[i].remove()
        plt_pnt = ax.scatter(c_x_co_ordinate,c_y_co_ordinate,s = 500, c= 'C' + str(i+2),marker='+',edgecolors = 'face' ,alpha  =1)
        culster_centroids_plot[i] = plt_pnt 
        time.sleep(.2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if (round(c_x_co_ordinate,1) == round(culster_centroids[i][0],1) and round(c_y_co_ordinate,1) == round(culster_centroids[i][1],1)):
            flag_cnt  += 1
        culster_centroids[i][0] = round(c_x_co_ordinate,1)
        culster_centroids[i][1] = round(c_y_co_ordinate,1)
    iter_num += 1
    if flag_cnt ==  c_num :
        break
plt.close(fig)
