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

dialog_train_data =[]
with open ('../../data/train_data_new_preprocessing_multi_answer', 'rb') as fp:
    dialog_train_data = pickle.load(fp)[0]

# 입력 포맷 변경
Dataset = []
for i, data in enumerate(dialog_train_data):
    Dataset.append(TaggedDocument(data[0],[str(data[1])]))

# training parameter setting
max_epochs = 100
vec_size = 400
alpha = 0.025


# setting doc2vec pre-traing env

model = Doc2Vec(alpha=alpha, min_alpha=0.0025, min_count=1, dm =1, workers=10)#vector_size=5, window=2, min_count=1, workers=4)
#dm=1 -> distributed memory method(preserve order)
#dm=0 -> DBOW(ignore order)
model.build_vocab(Dataset)

#hyper parameter tunning
methods = [0]#,1]
max_epochs_list = [200]#20,40,100, 200, 400,800,1600]
vec_size_list = [100]#,200,400,600,800]
window_size_list = [5]#,8,11]


# Adjust hyperparameter
for method in methods:
    for window_size in window_size_list:
        for vec_size in vec_size_list:
            for max_epoch in max_epochs_list:

                #load current training model
                '''
                if fname:
                    path = 'models'
                    files = os.listdir(pah)
                    for file_name in files:
                        if file_name
                        prev_tmp_file = get_tmpfile(fname)
                    fname.split('_')
                    fname[0]
                '''



                # training setting change (Hyper-Parameter)
                model = Doc2Vec(Dataset, window=window_size, vector_size=vec_size,  alpha=alpha, min_alpha=0.0025, min_count=2, dm=method,\
                                workers=12,epochs=max_epoch)

                print('Method %d Window_size %d  vec_size %d Max_epoch %d' %
                      (model.dm.__bool__(), window_size, vec_size, max_epoch))
                # training
                #model.train(Dataset, total_examples=model.corpus_count, epochs=model.iter)

                # decrease the learning rate
                #model.alpha -= 0.0002
                # fix the learning rate, no decay
                #model.min_alpha = model.alpha


                # Persis a model to disk(file name+method+winSize+vecSize+Max_epoch)
                fname = "../../models/new_preprocessing_multi_answer_deeptask_Sentence2vector_" \
                                    + str(int(model.dm.__bool__())) + '_'\
                                    + str(model.window)+'_'\
                                    + str(vec_size)+'_'\
                                    + str(max_epoch)

                model.save(fname)



