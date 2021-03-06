from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import Doc2Vec
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import threading



# Daemon thread 선언
sentence_set = np.load('data/sentence_set_grammerCheck_sentenceSplit.npy').item()
def searching_sentence(sentence_set):
    while (1):
        sentence_id = input("Input sentence ID : ")
        print(sentence_set[int(sentence_id)])
thread1 = threading.Thread(target=searching_sentence,args=(sentence_set,))
thread1.daemon = True
thread1.start()





#model loading
model = Doc2Vec.load("/home/bluish02-lab-linux/repository/sentence_vector/models/test3/s2v_1_8_400_800")
data = model.docvecs.doctag_syn0

# Run T-SNE algorithm
#tsne = TSNE(n_components=2)
#result = tsne.fit_transform(data)
#np.save('tsne_result',result)

result = np.load('tsne_result.npy')
# clustering

CLUSTER_N = 50
km = KMeans(n_clusters=CLUSTER_N)
km = km.fit(result)
labels = km.labels_


# visualize
N=CLUSTER_N
#matplotlib.rc('font', family='nanumgothic')
#matplotlib.rc('axes', unicode_minus=False)

# define the colormap
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap',cmaplist, N)

bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1)
'''
scat = ax.scatter(result[:,0],result[:,1],c=labels, cmap = cmap)
cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
cb.set_label('Labels')
ax.set_title('sentence_clusters')

plt.show()
'''
scaler = MinMaxScaler()
scaler.fit(result)
scaled_result = scaler.transform(result)


# loop through labels and plot each cluster
for label in set(labels):

    #loop through data points and plot
    same_label_points = []
    for i, data in enumerate(scaled_result):
        #add the data point as text
        if labels[i] == label:
            #plot the sentence point
            plt.annotate(str(i),data,horizontalalignment='center', verticalalignment='center',size=11,color=cmap(label))
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
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.show()