import sys, csv
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# input parameters
data_file = sys.argv[1]
dim1 = 0	# what SV dimensions to plot
dim2 = 1

# read data into a numpy matrix
data_matrix = np.array(list(csv.reader(open(data_file, "rb"), delimiter=","))).astype("float")

# create some labels for the data
feature_labels = []
for i in range(data_matrix.shape[1]):
	feature_labels.append("f" + str(i+1))
sv_labels = []
for i in range(data_matrix.shape[1]):
	sv_labels.append("sv" + str(i+1))
item_labels = []
for i in range(data_matrix.shape[1]):
	feature_labels.append("item" + str(i+1))

# heatmap of the raw data
fig1, ax1 = plt.subplots()
heatmap = ax1.pcolor(data_matrix)
ax1.set_yticks(np.arange(data_matrix.shape[0])+0.5, minor=False)
ax1.set_xticks(np.arange(data_matrix.shape[1])+0.5, minor=False)
ax1.set_xticklabels(item_labels, minor=False)
ax1.set_yticklabels(feature_labels, minor=False)


# compute the SVD
U, s, V = np.linalg.svd(data_matrix, full_matrices=True)
print s

# plot the singular values
fig3, ax3 = plt.subplots()
heatmap = ax3.pcolor(V)
ax3.set_yticks(np.arange(V.shape[0])+0.5, minor=False)
ax3.set_xticks(np.arange(V.shape[1])+0.5, minor=False)
ax3.set_xticklabels(sv_labels, minor=False)
ax3.set_yticklabels(sv_labels, minor=False)

fig2, ax2 = plt.subplots()
heatmap = ax2.pcolor(U)
ax2.set_yticks(np.arange(U.shape[0])+0.5, minor=False)
ax2.set_xticks(np.arange(U.shape[1])+0.5, minor=False)
ax2.set_xticklabels(sv_labels, minor=False)
ax2.set_yticklabels(item_labels, minor=False)

# plot the eigenvalues
plt4 = plt.plot(s)
plt.show()

# plot the specified singular values


# plot the cluster diagram
data_dist = pdist(data_matrix) # computing the distance
data_link = linkage(data_dist) # computing the linkage
dendrogram(data_link)
plt.xlabel('Items')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);
plt.show()
