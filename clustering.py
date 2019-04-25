from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from gensim.models.doc2vec import Doc2Vec
import codecs

#model load
model = Doc2Vec.load("/home/bluish02-lab-linux/repository/sentence_vector/models/test3/s2v_1_8_400_800")

# ndarray extract
data = model.docvecs.doctag_syn0




data = np.load('tsne_result.npy')




# find clustering using K-means algorithm
Sum_of_squared_distances = []
K = range(1,200)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()



