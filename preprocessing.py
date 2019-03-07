
''' DeepTask WoZ 취득 데이터 형태소 분석 '''
import json
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import ast
import itertools
import sys
sys.path.insert(0,'./utils/py-hanspell')
from hanspell import spell_checker
import re
import pickle



# KOALA NLP library load
from koalanlp.Util import initialize
from koalanlp.proc import Tagger
from koalanlp.proc import SentenceSplitter
from koalanlp import API


initialize(OKT='LATEST')#, HNN = 'LATEST', KKMA='LATEST' ,KMR='LATEST',EUNJEON='LATEST',ARIRANG='LATEST',
           #RHINO='LATEST',DAON='LATEST')  #: HNN=2.0.3
API_LIST = ['OKT']#, 'HNN', 'KKMA' ,'KMR','EUNJEON','ARIRANG', 'RHINO','DAON']
'''
splitter_1 = SentenceSplitter(API.OKT)
splitter_2 = SentenceSplitter(API.HNN)






# Split & grammery checked dialog save to file
with open('Deeptask_dialogDB.json', 'r') as fp:
    with open('Deeptask_dialogDB_splited_grammerChecked', 'wb') as wfp:

        wfp_lists = []
        for line_index, line in enumerate(fp):

            if line:
                loaded_json_data = json.loads(line)
            else:
                break

            dialog_id = line_index

            if not loaded_json_data['Naturalness']:
                continue
            else:
                # Score 계산
                #delta_score = np.abs(np.max(loaded_json_data['Naturalness']) - np.min(loaded_json_data['Naturalness']))
                average_score = np.mean(loaded_json_data['Naturalness'])

            dialog_sentence_list = []
            dialog_len = loaded_json_data['Dialog'].split('\n').__len__()
            dialog = loaded_json_data['Dialog']

            # Low score dialog filtering
            if average_score < 2.5:
                continue

            # To seeing progress
            print('processing... ', line_index)

            for line in spell_checker.check(dialog.split('\n')):

                dialog_string_role_removed = ''.join(str(elem) for elem in re.split('[:]',line.checked)[1:]).strip()
                paragraph1 = splitter_1(dialog_string_role_removed)
                #paragraph2 = splitter_2(dialog_string_role_removed)

                #if paragraph1.__len__() != paragraph2.__len__():
                #    continue
                #else:
                    # sentence write
                #for sentence in paragraph1:

                dialog_sentence_list.extend(paragraph1)

            wfp_lists.append([dialog_sentence_list,dialog_id])

        pickle.dump(wfp_lists,wfp)
'''



# Tokenize (POS Tagging)
with open ('sentence_set_grammerCheck_sentenceSplit.npy', 'rb') as fp:
    #data = pickle.load(fp)
    data = np.load(fp)

#initialize(OKT='LATEST', HNN = 'LATEST', KKMA='LATEST' ,KMR='LATEST',EUNJEON='LATEST',ARIRANG='LATEST',
#           RHINO='LATEST',DAON='LATEST', ETRI='LATEST')

for api in API_LIST:
    tagger = Tagger(api.lower())
#tagger2 = Tagger(API.HNN)
#tagger3 = Tagger(API.KKMA)
#tagger4 = Tagger(API.KMR)
#tagger5 = Tagger(API.EUNJEON)
#tagger6 = Tagger(API.ARIRANG)
#tagger7 = Tagger(API.RHINO)
#tagger8 = Tagger(API.DAON)

    lists = []
    for key, value in data.item().items():
        print('processing',key)
        taggedParagaph1 = tagger(value)
        #taggedParagaph2 = tagger2(dialog[0])
        #taggedParagaph3 = tagger3(dialog[0])
        #taggedParagaph4 = tagger4(dialog[0])
        #taggedParagaph5 = tagger5(dialog[0])
        #taggedParagaph6 = tagger6(dialog[0])
        #taggedParagaph7 = tagger7(dialog[0])
        #taggedParagaph8 = tagger8(dialog[0])

        splited_sentence = re.split('[+ ]', taggedParagaph1[0].singleLineString())
        lists.append(splited_sentence)

    #save tokenized data to file
    with open('tokenized_sentence_set_'+api,'wb') as wfp:
        pickle.dump(lists,wfp)











'''
# Spellcheck and Sentence Separation 
with open('Deeptask_dialogDB.json', 'r') as fp:
    for line_index, line in enumerate(fp):
        if line:
            loaded_json_data = json.loads(line)
        else:
            break

        dialog_id = line_index

        if not loaded_json_data['Naturalness']:
            continue
        else:
            # Score 계산
            #delta_score = np.abs(np.max(loaded_json_data['Naturalness']) - np.min(loaded_json_data['Naturalness']))
            average_score = np.mean(loaded_json_data['Naturalness'])


        dialog_len = loaded_json_data['Dialog'].split('\n').__len__()
        dialog = loaded_json_data['Dialog']

        # Low score dialog filtering
        if average_score < 2.5:
            continue

        # To seeing progress
        print('processing... ', line_index)

        for line in spell_checker.check(dialog.split('\n')):

            dialog_string_role_removed = ''.join(str(elem) for elem in re.split('[:]',line.checked)[1:]).strip()
            paragraph1 = splitter_1(dialog_string_role_removed)
            #paragraph2 = splitter_2(dialog_string_role_removed)

            #if paragraph1.__len__() != paragraph2.__len__():
            #    continue
            #else:
                # sentence write
            for sentence in paragraph1:

                taggedSentence = tagger(sentence)
                tokenized_sentence = []


                sentence_list.append(sentence)
                row_index += 1

'''








