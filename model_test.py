import nltk
import numpy as np

from pprint import pprint
import pickle
import random

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



def column(matrix, i):
    return [row[i] for row in matrix]




#load pre-trained data(kor)
#pre_trained_model = gensim.models.Word2Vec.load('data/ko.bin')




#load test data
dialog_train_data =[]
with open ('../../data/train_data_group2_only', 'rb') as fp:
    dialog_train_data = pickle.load(fp)[0]



#Analysis & inferrence
#hyper parameter list
methods = [0]#,1]
max_epochs_list = [200]#20,40,100, 200, 400,800,1600]
vec_size_list = [100]#,200,400,600,800]
window_size_list = [5]#,8,11]

test_cnt_list = []
for method in methods:
    for window_size in window_size_list:
        for vec_size in vec_size_list:
            for max_epoch in max_epochs_list:
                model = Doc2Vec.load("../../models/G2_deeptask_Sentence2vector_"
                                     + str(method) + '_'
                                     + str(window_size) + '_'
                                     + str(vec_size) + '_'
                                     + str(max_epoch)
                                     )

                # Top influence word print for Analysis
                # label 별 통계 정보 표시
                for label in sorted(model.docvecs.doctags):
                    print(model.docvecs.doctags[label])
                for label in sorted(model.docvecs.doctags):
                    doc_vec = model.docvecs[label]
                    print(label)
                    print(model.most_similar([doc_vec],topn=20))
                #print('Max_epoch %d vec_size %d ' % (max_epoch, vec_size), model.most_similar(positive=['감사해요.'], topn=3))
                print('Method %d Window_size %d  vec_size %d Max_epoch %d ' %
                      (method, window_size, vec_size, max_epoch))

                cnt = 0
                unmatch_cnt = 0
                for test_string in random.sample(column(dialog_train_data,2),dialog_train_data.__len__()):

                    inference_sentence = test_string
                    tagged_id = str([id for tokens ,id, string in dialog_train_data if string == inference_sentence][0])
                    tokens = [tokens for tokens, id, string in dialog_train_data if string == inference_sentence][0]
                    string = [string for tokens, id, string in dialog_train_data if string == inference_sentence][0]
                    infered_vector = model.infer_vector(tokens)
                    similarities = model.docvecs.most_similar([infered_vector],topn=len(model.docvecs))
                    rank = [category for category, similarity in similarities]


                    if rank[0] == str(tagged_id):
                        cnt += 1
                        #print(string, tagged_id, rank)
                    else:
                        unmatch_cnt += 1
                        print(string, tagged_id, rank[0])
                print('total_len',dialog_train_data.__len__(),'cnt',cnt, 'unmatch_cnt',unmatch_cnt)
                test_cnt_list.append([method, window_size, vec_size, max_epoch,cnt/dialog_train_data.__len__()])




                #Infer word

                for key in model.wv.vocab.keys():
                    if key.startswith('친구'):
                        print(key)
                        print(model.similar_by_word(key, topn=5))

                print('\n')
for data in sorted(test_cnt_list, key=lambda x:x[4]):
    print(data)


#flatten_dialogs = [item for sublist in dialog_train_data for item in sublist[0]]
#dataset_nltk = nltk.Text(flatten_dialogs,'DeepTasK')

#model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
#model = doc2vec.load(fname)
#vector = model.infer_vector(["system", "response"])





