import nltk
import numpy as np

from pprint import pprint
import pickle

from collections import namedtuple

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import gensim

from sklearn.linear_model import LogisticRegression

from matplotlib import font_manager, rc
font_fname = '/usr/share/fonts/NanumFont/NanumGothic.ttf'     # A font of your choice
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)


# Data loading to train
'''
dialog_train_data =[]
with open ('Deeptask_dialogDB_splited_grammerChecked', 'rb') as fp:
    dialog_train_data = pickle.load(fp)
'''


# Sentecne-set load
sentence_set = np.load('sentence_set_grammerCheck_sentenceSplit.npy').item()

# doc2vec data format construction
# first try
#dialogs = [TaggedDocument(data[0], str(data[1]))  for data in dialog_train_data]

# second try
'''
Dataset = []

for data in dialog_train_data:
    dialog = []
    for sentence in data[0]:
        # Word
        words = sentence.split()
        Dataset.append(TaggedDocument(words,str([id for id,string  in sentence_set.items() if string == sentence][0])))
        # Sentence
        #Dataset.append(TaggedDocument(sentence,str(data[1])))


    #Dataset.append(TaggedDocument(dialog,str(data[1])))
'''
# Third try
'''
Dataset = []
for id, sentence in sentence_set.items():
    words = sentence.split()
    Dataset.append(TaggedDocument(words, str(id)))
'''
# Fourth try - Load tokenized sentence set
with open ('tokenized_sentence_set_OKT', 'rb') as fp:
    dialog_train_data = pickle.load(fp)

Dataset = []
for i, data in enumerate(dialog_train_data):
    Dataset.append(TaggedDocument(data,[sentence_set[i]]))

# training parameter setting
max_epochs = 100
vec_size = 400

alpha = 0.025


# setting doc2vec pre-traing env
'''
model = Doc2Vec(alpha=alpha, min_alpha=0.0025, min_count=1, dm =1, workers=10)#vector_size=5, window=2, min_count=1, workers=4)
#dm=1 -> distributed memory method(preserve order)
#dm=0 -> DBOW(ignore order)
model.build_vocab(Dataset)

#hyper parameter tunning
max_epochs_list = [100, 200, 400,800]
vec_size_list = [400]
window_size_list = [8]
#for method in [0, 1]:

# Adjust hyperparameter
for window_size in window_size_list:
    for vec_size in vec_size_list:
        for max_epoch in max_epochs_list:

            #load current training model
            #if fname:
            #    fname.split('_')
            for epoch in range(max_epoch):
                print('Method %d Window_size %d  vec_size %d Max_epoch %d - epoch %d' %
                      (model.dm.__bool__(),window_size ,vec_size, max_epoch, epoch))

                # training setting change (Hyper-Parameter)
                model.window = window_size
                model.vector_size = vec_size

                # training
                model.train(Dataset, total_examples=model.corpus_count, epochs=model.iter)

                # decrease the learning rate
                model.alpha -= 0.0002
                # fix the learning rate, no decay
                model.min_alpha = model.alpha


            # Persis a model to disk(file name+method+winSize+vecSize+Max_epoch)
            fname = get_tmpfile("/home/bluish02-lab-linux/repository/deeptask_woz/doc2vec_models/test3/s2v_"
                                + str(int(model.dm.__bool__())) + '_'
                                + str(model.window)+'_'
                                + str(vec_size)+'_'
                                + str(max_epoch)
                                )
            model.save(fname)


'''



#Analysis & inferrence
#hyper parameter list
max_epochs_list = [100, 200, 400,800]
vec_size_list = [400]
window_size_list = [8]
for window_size in window_size_list:
    for vec_size in vec_size_list:
        for max_epoch in max_epochs_list:
            model = Doc2Vec.load("/home/bluish02-lab-linux/repository/deeptask_woz/doc2vec_models/test3/s2v_"
                                 + '1' + '_'
                                 + str(window_size) + '_'
                                 + str(vec_size) + '_'
                                 + str(max_epoch)
                                 )
            #print('Max_epoch %d vec_size %d ' % (max_epoch, vec_size), model.most_similar(positive=['감사해요.'], topn=3))
            print('Method %d Window_size %d  vec_size %d Max_epoch %d ' %
                  (1, window_size, vec_size, max_epoch))

            inference_sentence = ''
            sentence_id = str([id for id, string in sentence_set.items() if string == inference_sentence][0])
            model.infer_vector(inference_sentence.split())
            model.docvecs.most_similar(sentence_id)

#flatten_dialogs = [item for sublist in dialog_train_data for item in sublist[0]]
#dataset_nltk = nltk.Text(flatten_dialogs,'DeepTasK')

#model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
#model = doc2vec.load(fname)
#vector = model.infer_vector(["system", "response"])



