# clustering

CLUSTER_N = 10
km = KMeans(n_clusters=CLUSTER_N)
km = km.fit(result)
labels = km.labels_

# visualize
N = CLUSTER_N
mpl.rc('font', family='nanumgothic')
mpl.rc('axes', unicode_minus=False)

# define the colormap
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, N)

bounds = np.linspace(0, N, N + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1, 1)
'''
scat = ax.scatter(result[:,0],result[:,1],c=labels, cmap = cmap)
cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
cb.set_label('Labels')
ax.set_title('sentence_clusters')

plt.show()
'''
scaler = MinMaxScaler()
concatenated_result = np.concatenate((result,result_docvecs),axis=0)
scaler.fit(concatenated_result)
scaled_result = scaler.transform(result)

# loop through labels and plot each cluster
for label in set(labels):

    # loop through data points and plot
    same_label_points = []
    for i, data in enumerate(scaled_result):
        # add the data point as text
        if labels[i] == label:
            # plot the sentence point
            plt.annotate(str(i)+'('+str(dialog_train_data[i][1])+')', data, horizontalalignment='center', verticalalignment='center', size=11,
                         color=cmap(label))
            same_label_points.append(data)

        else:
            continue

    # Plot the decision boundary
    '''
    h = 0.02 # Step size of the mesh. Decrease to increase the quality of the VQ.
    np_same_label_points = np.asarray(same_label_points)
    x_min, x_max = np_same_label_points[:,0].min() -1, np_same_label_points[:,0].max() +1
    y_min, y_max = np_same_label_points[:, 1].min() - 1, np_same_label_points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h),np.arange(y_min,y_max,h))

    Z = km.predict(scaler.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(),yy.min(), yy.max()),
               cmap=cmap,
               aspect='auto', origin='lower',
               alpha=0.2
               )
    '''
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
#plt.show()


# clustering

CLUSTER_N = 10
km = KMeans(n_clusters=CLUSTER_N)
km = km.fit(result_docvecs)
labels = km.labels_

# visualize
N = CLUSTER_N
# matplotlib.rc('font', family='nanumgothic')
# matplotlib.rc('axes', unicode_minus=False)

# define the colormap
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, N)

bounds = np.linspace(0, N, N + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#fig, ax = plt.subplots(1, 1)
'''
scat = ax.scatter(result[:,0],result[:,1],c=labels, cmap = cmap)
cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
cb.set_label('Labels')
ax.set_title('sentence_clusters')

plt.show()
'''
#scaler = MinMaxScaler()
#scaler.fit(result_docvecs)
scaled_result = scaler.transform(result_docvecs)

# loop through labels and plot each cluster
for label in set(labels):

    # loop through data points and plot
    same_label_points = []
    for i, data in enumerate(scaled_result):
        # add the data point as text
        if labels[i] == label:
            # plot the sentence point
            docvecs_tag = model.docvecs.offset2doctag[i]
            plt.annotate(docvecs_tag, data, horizontalalignment='center', verticalalignment='center', size=50,
                         color=cmap(label))
            same_label_points.append(data)

        else:
            continue

    # Plot the decision boundary
    '''
    h = 0.02 # Step size of the mesh. Decrease to increase the quality of the VQ.
    np_same_label_points = np.asarray(same_label_points)
    x_min, x_max = np_same_label_points[:,0].min() -1, np_same_label_points[:,0].max() +1
    y_min, y_max = np_same_label_points[:, 1].min() - 1, np_same_label_points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h),np.arange(y_min,y_max,h))

    Z = km.predict(scaler.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(),yy.min(), yy.max()),
               cmap=cmap,
               aspect='auto', origin='lower',
               alpha=0.2
               )
    '''
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.show()








