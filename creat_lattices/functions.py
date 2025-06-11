# -*- coding: utf-8 -*-

import sys
import numpy as np
import re
from creat_lattices.alphabet import Alphabet
# from transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer

NULLKEY = "-null-"

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(num_layer, input_file, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    
    global w_length,words,matched_list,matched_length,matched_Id,g,c, gaz_chars,gaz_Ids,gazs,gazs_count,gaz_char_Id, idx, label, label_len, count_set, mask, gazmask,gazcharmask,layergazmasks,gazchar_masks,texts,bert_text_ids,instence_texts,instence_Ids
    
    # print(num_layer)
    # print(input_file)
    # print(gaz)
    # print(word_alphabet)
    # print(biword_alphabet)
    # print(biword_count)
    # print(char_alphabet)
    # print(gaz_alphabet) 
    # print(gaz_count)
    # print(gaz_split)
    # print(label_alphabet)
    # print( number_normalized)
    # print(max_sent_length)

    tokenizer = BertTokenizer.from_pretrained('medbert-base-wwm-chinese', do_lower_case=True)

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)): #每个样本
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)                         #biwords
            words.append(word)                              #words
            labels.append(label)                            #labels 每个字的label
            word_Ids.append(word_alphabet.get_index(word))  #word_Ids
            biword_index = biword_alphabet.get_index(biword)
            biword_Ids.append(biword_index)                 #biword_Ids
            label_Ids.append(label_alphabet.get_index(label))  #label_Ids
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0: 
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else: #char_padding_size=-1
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)                           #char
            char_Ids.append(char_Id)                          #char_Ids

        else: #两个样本间的空行
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)

                gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                # print(gazs)  #[[[], [], [], []], [[], [], [], []], ..., [[], [], [], []]] 17个
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

                max_gazlist = 0
                max_gazcharlen = 0
                # print(w_length)  #17
                for idx in range(w_length): #以句子中的每个字为开头   
                    # print(words[idx]) #高
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    # print(matched_list) 
                    #以该字words[idx]为开头的所有词（包括字本身） ['高勇'，'高']
                    matched_length = [len(a) for a in matched_list]                     
                    # print(matched_length)
                    #以该字words[idx]为开头所有词的长度列表 [2, 1]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    # print(matched_Id)
                    #以该字words[idx]为开头的所有词在gaz_alphabet中的id列表 [2, 3]

                    if matched_length:
                        max_gazcharlen = max(max(matched_length),max_gazcharlen)
                        #以该字words[idx]为开头所有词的最大长度

                    for w in range(len(matched_Id)):#以该字words[idx]为开头的每个词的id
                        gaz_chars = []
                        g = matched_list[w] #以该字words[idx]为开头的某词g（包括字本身）
                        # print(g) #第一个词：高勇    第二个词：高
                        for c in g: #某词g中的每个字c
                            # print(c) #第一个词：高  勇   第二个词：高
                            gaz_chars.append(word_alphabet.get_index(c)) 
                            #每个字c在word_alphabet中的id加入gaz_chars
                        # print(gaz_chars) #第一个词：[2, 3] 第二个词：[2]
                        
                        # print(matched_length[w])  第一个词：2  第二个词：1
                        if matched_length[w] == 1: #Single 如果以该字words[idx]为开头的某词g是单字 
                            #第二个词： ‘高’
                            # print(matched_Id[w]) #3 ‘高’在 gaz_alphabet中的id
                            gazs[idx][3].append(matched_Id[w]) 
                            #将词g在gaz_alphabet中的id加入该字words[idx]的gazs的第四列（标签S）
                            gazs_count[idx][3].append(1)
                            # print(gaz_chars) #[2]
                            gaz_char_Id[idx][3].append(gaz_chars)
                            #将词g每个字c在word_alphabet中的id加入words[idx]的gaz_char_Id的第四列
                        
                        else:   #如果以该字words[idx]为开头的某词g不是单字  第一个词：‘高勇’
                            # print(matched_Id[w]) #2   ‘高勇’在 gaz_alphabet中的id
                            gazs[idx][0].append(matched_Id[w])   ## Begin
                            #词g在gaz_alphabet的id加入该字words[idx]的gazs第一列（标签B）
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                            # print(gaz_chars) # [2,3] 分别代表‘高’、‘勇’在word_alphabet中的id
                            gaz_char_Id[idx][0].append(gaz_chars)
                            #将词g每个字c在word_alphabet中的id加入words[idx]的gaz_char_Id的第一列
                            wlen = matched_length[w] #词g的长度
                            gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                            #词g在gaz_alphabet中的id加入前面的字words[idx+wlen-1]的gazs第3列（标签E）
                            gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                            #将词g每个字c在word_alphabet中的id加入words[idx+wlen-1]的gaz_char_Id的第3列
                            for l in range(wlen-2): #如果词g的长度大于2
                                gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                                #词g在gaz_alphabet的id加入前面的字words[idx+l-1]的gazs第2列（标签M）
                                gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)
                                #将词g每个字c在word_alphabet中的id加入words[idx+l-1]的gaz_char_Id的第2列

                    for label in range(4):
                        if not gazs[idx][label]: #如果该字words[idx]的gazs的某标签列没有词，用0代替
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                        max_gazlist = max(len(gazs[idx][label]),max_gazlist) #BMES包含的单词数的最大值
                    # print(max_gazlist) #1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2
                    
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    #以该字words[idx]为开头的所有词在gaz_alphabet中的id列表（前面不是有吗）
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                        #以该字words[idx]为开头的所有词在gaz_alphabet中的id及其长度列表，放入gaz_Ids中，后面words[idx+1]...不断累积
                    else:
                        gaz_Ids.append([])
                
                # print(max_gazlist)                      
                # print(len(gaz_Ids)) #17 第一句 
                #'高'对应的词
                # print(gaz_Ids[0]) # [[2, 3], [2, 1]] 分别代表['高勇','高']在gaz_alphabet中的id，和长度
                # print(len(gazs)) #17
                #'高'对应的词划分成BMES后
                # print(gazs[0]) #[[2], [0], [0], [3]] 分别代表['高勇',‘NONE’,‘NONE’,'高'] 在gaz_alphabet中的id
                # print(len(gazs_count)) #17
                # print(gazs_count[0]) #[[1], [1], [1], [1]] BMES分别有一个词
                # print(len(gaz_char_Id)) #17
                # print(gaz_char_Id[0]) #[[[2, 3]], [[0]], [[0]], [[2]]] 分别代表['高勇',‘NONE’,‘NONE’,'高']的每个字在word_alphabet中的id

                ## batch_size = 1  
                # print(w_length)  #17 
                for idx in range(w_length): 
                    # print(words[idx]) #高
                    gazmask = []
                    gazcharmask = []
                    # print('idx:',idx )#idx: 0
                    for label in range(4): #对于B M E S每个标签
                        # print('label:',label) #label: 0  label: 1  label: 2  label: 3
                        # print(gazs[idx][label]) #‘高’B,即'高勇'的id 2
                        label_len = len(gazs[idx][label]) 
                        #words[idx]的某标签包含的单词数 最少为1，因为前面已经用0填充了
                        # print(label_len) #‘高’B的：1  M:1  E:1  S:1
                        count_set = set(gazs_count[idx][label]) #label_len的集合
                        # print(count_set) #‘高’B的：{1} M:{1} E:{1} S:{1}
                        if len(count_set) == 1 and 0 in count_set: #如果前面没有用0填充
                            gazs_count[idx][label] = [1]*label_len
                            
                        #使整个句子BMES包含的单词数一样，不够的用0填充
                        mask = label_len*[0]  #真实单词的mask
                        # print(max_gazlist) #第一句中BMES包含的单词数的最大值 2
                        mask += (max_gazlist-label_len)*[1] #填充部分的mask
                        # print(mask) #‘高’B的：[0, 1] M:[0, 1] E:[0, 1] S:[0, 1]
                        
                        gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding 不够的用0填充
                        # print(gazs[idx][label]) # word[idx]某label下的单词在gaz_alphabet中的id 
                        #‘高’B的：[2, 0] M:[0, 0] E:[0, 0] S:[3, 0]                        
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding 不够的用0填充
                        # print(gazs_count[idx][label])#word[idx]某label下的单词的数量 
                        #‘高’B的：[1, 0]  M:[1, 0] E:[1, 0] S:[1, 0]

                        #使整个句子BMES单词的字数相同，不够的用0填充
                        # print(max_gazcharlen)  #3 第一句对应的所有词的最大长度
                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):#word[idx]某label下的每个单词
                            glen = len(gaz_char_Id[idx][label][g])
                            #word[idx]某label下的第g个单词有几个字
                            charmask = glen*[0] #真实单词的mask为0
                            charmask += (max_gazcharlen-glen) * [1]#填充部分mask为1
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        
                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]] #max_gazcharlen：第一句对应的所有词的最大长度  #max_gazlist：第一句BMES包含的单词数的最大值 
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                        gazmask.append(mask)
                        # print(gazmask)#“高”每个label的单词数填充mask
                        #累积 ‘高’B的：[[0, 1]] M:[[0, 1], [0, 1]] E:[[0, 1], [0, 1], [0, 1]] S:[[0, 1], [0, 1], [0, 1], [0, 1]]
                        gazcharmask.append(char_mask)
                        # print(gazcharmask)#“高”每个label的每个单词长度填充mask 
                        #累积 ‘高’B的：[[[0, 0, 1], [1, 1, 1]]] M:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]  E:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]  S:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]
                    layergazmasks.append(gazmask)
                    # print(layergazmasks) #“高”所有label的
                    #[[[0, 1], [0, 1], [0, 1], [0, 1]]]
                    gazchar_masks.append(gazcharmask)
                    # print(gazchar_masks)#“高”所有label的
                    #[[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]]

                texts = ['[CLS]'] + words + ['[SEP]'] #第一句的text
                # print(texts) #第一句  ['[CLS]', '高', '勇', '：', '男', '，', '中', '国', '国', '籍', '，', '无', '境', '外', '居', '留', '权', '，', '[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)
                # print(bert_text_ids) #第一句 [101, 7770, 1235, 8038, 4511, 8024, 704, 1744, 1744, 5093, 8024, 3187, 1862, 1912, 2233, 4522, 3326, 8024, 102]
                instence_texts.append([words, biwords, chars, gazs, labels])
                # print(instence_texts) #第一句
                # [[words ['高', '勇', '：', '男', '，', '中', '国', '国', '籍', '，', '无', '境', '外', '居', '留', '权', '，'], 
                # biwords ['高勇', '勇：', '：男', '男，', '，中', '中国', '国国', '国籍', '籍，', '，无', '无境', '境外', '外居', '居留', '留权', '权，', '，-null-'], 
                 # chars [['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']],
                # gazs 各字对应的BWES单词在gaz_alphabet中的id [[[2, 0], [0, 0], [0, 0], [3, 0]], [[0, 0], [0, 0], [2, 0], [4, 0]],...,[[0, 0], [0, 0], [16, 0], [20, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]], 
                #labels ['B-NAME', 'E-NAME', 'O', 'O', 'O', 'B-CONT', 'M-CONT', 'M-CONT', 'E-CONT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]]
                
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids])
                # print(instence_Ids) #第一句
                # [word_Ids [[2, 3, 4, 5, 6, 7, 8, 8, 9, 6, 10, 11, 12, 13, 14, 15, 6],
                #  biword_Ids [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
                  # char_Ids [[2], [3], [4], [5], [6], [7], [8], [8], [9], [6], [10], [11], [12], [13], [14], [15], [6]],
                  # gaz_Ids 各字的匹配单词(如['高勇'，'高']在gaz_alphabet中的id(如[2, 3])和长度(如[2, 1])  [[[2, 3], [2, 1]], [[4], [1]], [], ..., [3, 2, 1]], [[19], [1]], [[20], [1]], []], 
                  # label_Ids [1, 2, 3, 3, 3, 4, 5, 5, 6, 3, 3, 3, 3, 3, 3, 3, 3], 
                   # gazs 各字的BMES单词在gaz_alphabet中的id([[[2, 0], [0, 0], [0, 0], [3, 0]], [[0, 0], [0, 0], [2, 0], [4, 0]],...,[[0, 0], [0, 0], [16, 0], [20, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]],
                  # gazs_count [[[1, 0], [1, 0], [1, 0], [1, 0]], [[1, 0], [1, 0], [1, 0], [1, 0]], ..., [[1, 0], [1, 0], [185, 0], [1, 0]], [[1, 0], [1, 0], [1, 0], [1, 0]]],
                  # gaz_char_Id 各字的BMES单词的字在word_alphabet中的id [[[[2, 3, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[2, 3, 0], [0, 0, 0]], [[3, 0, 0], [0, 0, 0]]], ..., [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[13, 14, 15], [0, 0, 0]], [[15, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]], 
                  # layergazmasks 各字的BMES单词数都是两个，第一个是原本的（包含第一次填充的‘NONE’）mask为0，第二个是第二次填充的mask为1 [[[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]], ..., [[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]]], 
                  #gazchar_masks 各字的BMES单词的字数都是3个 [[B[[0‘高’, 0‘勇’, 1], [1, 1, 1]], M[[0‘NONE’, 1, 1], [1, 1, 1]], E[[0‘NONE’, 1, 1], [1, 1, 1]], S[[0‘勇’, 1, 1], [1, 1, 1]]], [[[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]], ..., [B[[0, 1, 1], [1, 1, 1]], M[[0, 1, 1], [1, 1, 1]], E[[0'居', 0'留', 0'、权'], [1, 1, 1]], S[[0, 1, 1], [1, 1, 1]]], [[[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]], 
                  # bert_text_ids 各字在bert词典中的id [101, 7770, 1235, 8038, 4511, 8024, 704, 1744, 1744, 5093, 8024, 3187, 1862, 1912, 2233, 4522, 3326, 8024, 102]]]

            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids

def read_instance_with_gaz_yidu(num_layer, input_file, gaz, word_alphabet,  char_alphabet, gaz_alphabet, gaz_count, gaz_split,  number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    
    global w_length,words,matched_list,matched_length,matched_Id,g,c, gaz_chars,gaz_Ids,gazs,gazs_count,gaz_char_Id, idx, label, label_len, count_set, mask, gazmask,gazcharmask,layergazmasks,gazchar_masks,texts,bert_text_ids,instence_texts,instence_Ids

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    in_lines= pickle.load(open(input_file, 'rb'))
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for item in in_lines.items():
        key = item[0]
        # print(key)
        value = item[1]
        for word in key:
            # print(word)
            if number_normalized:  #True
                word = normalize_word(word)
                # print(word) 
            words.append(word)
        # print(word_list)
        for words in value:
            for word in words:
                # print(word)
                if number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) 
                words.append(word)
                # print(word_list)

            word_Ids.append(word_alphabet.get_index(word))  #word_Ids
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0: 
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else: #char_padding_size=-1
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)                           #char
            char_Ids.append(char_Id)                          #char_Ids

        
        if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
            gaz_Ids = []
            layergazmasks = []
            gazchar_masks = []
            w_length = len(words)

            gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
            # print(gazs)  #[[[], [], [], []], [[], [], [], []], ..., [[], [], [], []]] 17个
            gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
            gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

            max_gazlist = 0
            max_gazcharlen = 0
            # print(w_length)  #17
            for idx in range(w_length): #以句子中的每个字为开头   
                # print(words[idx]) #高
                matched_list = gaz.enumerateMatchList(words[idx:])
                # print(matched_list) 
                #以该字words[idx]为开头的所有词（包括字本身） ['高勇'，'高']
                matched_length = [len(a) for a in matched_list]                     
                # print(matched_length)
                #以该字words[idx]为开头所有词的长度列表 [2, 1]
                matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
                # print(matched_Id)
                #以该字words[idx]为开头的所有词在gaz_alphabet中的id列表 [2, 3]

                if matched_length:
                    max_gazcharlen = max(max(matched_length),max_gazcharlen)
                    #以该字words[idx]为开头所有词的最大长度

                for w in range(len(matched_Id)):#以该字words[idx]为开头的每个词的id
                    gaz_chars = []
                    g = matched_list[w] #以该字words[idx]为开头的某词g（包括字本身）
                    # print(g) #第一个词：高勇    第二个词：高
                    for c in g: #某词g中的每个字c
                        # print(c) #第一个词：高  勇   第二个词：高
                        gaz_chars.append(word_alphabet.get_index(c)) 
                        #每个字c在word_alphabet中的id加入gaz_chars
                    # print(gaz_chars) #第一个词：[2, 3] 第二个词：[2]
                        
                    # print(matched_length[w])  第一个词：2  第二个词：1
                    if matched_length[w] == 1: #Single 如果以该字words[idx]为开头的某词g是单字 
                        #第二个词： ‘高’
                        # print(matched_Id[w]) #3 ‘高’在 gaz_alphabet中的id
                        gazs[idx][3].append(matched_Id[w]) 
                        #将词g在gaz_alphabet中的id加入该字words[idx]的gazs的第四列（标签S）
                        gazs_count[idx][3].append(1)
                        # print(gaz_chars) #[2]
                        gaz_char_Id[idx][3].append(gaz_chars)
                        #将词g每个字c在word_alphabet中的id加入words[idx]的gaz_char_Id的第四列
                        
                    else:   #如果以该字words[idx]为开头的某词g不是单字  第一个词：‘高勇’
                        # print(matched_Id[w]) #2   ‘高勇’在 gaz_alphabet中的id
                        gazs[idx][0].append(matched_Id[w])   ## Begin
                        #词g在gaz_alphabet的id加入该字words[idx]的gazs第一列（标签B）
                        gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                        # print(gaz_chars) # [2,3] 分别代表‘高’、‘勇’在word_alphabet中的id
                        gaz_char_Id[idx][0].append(gaz_chars)
                        #将词g每个字c在word_alphabet中的id加入words[idx]的gaz_char_Id的第一列
                        wlen = matched_length[w] #词g的长度
                        gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                        #词g在gaz_alphabet中的id加入前面的字words[idx+wlen-1]的gazs第3列（标签E）
                        gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                        gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                        #将词g每个字c在word_alphabet中的id加入words[idx+wlen-1]的gaz_char_Id的第3列
                        for l in range(wlen-2): #如果词g的长度大于2
                            gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                            #词g在gaz_alphabet的id加入前面的字words[idx+l-1]的gazs第2列（标签M）
                            gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+l+1][1].append(gaz_chars)
                            #将词g每个字c在word_alphabet中的id加入words[idx+l-1]的gaz_char_Id的第2列

                for label in range(4):
                    if not gazs[idx][label]: #如果该字words[idx]的gazs的某标签列没有词，用0代替
                        gazs[idx][label].append(0)
                        gazs_count[idx][label].append(1)
                        gaz_char_Id[idx][label].append([0])

                    max_gazlist = max(len(gazs[idx][label]),max_gazlist) #BMES包含的单词数的最大值
                # print(max_gazlist) #1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2
                    
                matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                #以该字words[idx]为开头的所有词在gaz_alphabet中的id列表（前面不是有吗）
                if matched_Id:
                    gaz_Ids.append([matched_Id, matched_length])
                    #以该字words[idx]为开头的所有词在gaz_alphabet中的id及其长度列表，放入gaz_Ids中，后面words[idx+1]...不断累积
                else:
                    gaz_Ids.append([])
                
            # print(max_gazlist)                      
            # print(len(gaz_Ids)) #17 第一句 
            #'高'对应的词
            # print(gaz_Ids[0]) # [[2, 3], [2, 1]] 分别代表['高勇','高']在gaz_alphabet中的id，和长度
            # print(len(gazs)) #17
            #'高'对应的词划分成BMES后
            # print(gazs[0]) #[[2], [0], [0], [3]] 分别代表['高勇',‘NONE’,‘NONE’,'高'] 在gaz_alphabet中的id
            # print(len(gazs_count)) #17
            # print(gazs_count[0]) #[[1], [1], [1], [1]] BMES分别有一个词
            # print(len(gaz_char_Id)) #17
            # print(gaz_char_Id[0]) #[[[2, 3]], [[0]], [[0]], [[2]]] 分别代表['高勇',‘NONE’,‘NONE’,'高']的每个字在word_alphabet中的id

            ## batch_size = 1  
            # print(w_length)  #17 
            for idx in range(w_length): 
                # print(words[idx]) #高
                gazmask = []
                gazcharmask = []
                # print('idx:',idx )#idx: 0
                for label in range(4): #对于B M E S每个标签
                    # print('label:',label) #label: 0  label: 1  label: 2  label: 3
                    # print(gazs[idx][label]) #‘高’B,即'高勇'的id 2
                    label_len = len(gazs[idx][label]) 
                    #words[idx]的某标签包含的单词数 最少为1，因为前面已经用0填充了
                    # print(label_len) #‘高’B的：1  M:1  E:1  S:1
                    count_set = set(gazs_count[idx][label]) #label_len的集合
                    # print(count_set) #‘高’B的：{1} M:{1} E:{1} S:{1}
                    if len(count_set) == 1 and 0 in count_set: #如果前面没有用0填充
                        gazs_count[idx][label] = [1]*label_len
                            
                    #使整个句子BMES包含的单词数一样，不够的用0填充
                    mask = label_len*[0]  #真实单词的mask
                    # print(max_gazlist) #第一句中BMES包含的单词数的最大值 2
                    mask += (max_gazlist-label_len)*[1] #填充部分的mask
                    # print(mask) #‘高’B的：[0, 1] M:[0, 1] E:[0, 1] S:[0, 1]
                        
                    gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding 不够的用0填充
                    # print(gazs[idx][label]) # word[idx]某label下的单词在gaz_alphabet中的id 
                    #‘高’B的：[2, 0] M:[0, 0] E:[0, 0] S:[3, 0]                        
                    gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding 不够的用0填充
                    # print(gazs_count[idx][label])#word[idx]某label下的单词的数量 
                    #‘高’B的：[1, 0]  M:[1, 0] E:[1, 0] S:[1, 0]

                    #使整个句子BMES单词的字数相同，不够的用0填充
                    # print(max_gazcharlen)  #3 第一句对应的所有词的最大长度
                    char_mask = []
                    for g in range(len(gaz_char_Id[idx][label])):#word[idx]某label下的每个单词
                        glen = len(gaz_char_Id[idx][label][g])
                        #word[idx]某label下的第g个单词有几个字
                        charmask = glen*[0] #真实单词的mask为0
                        charmask += (max_gazcharlen-glen) * [1]#填充部分mask为1
                        char_mask.append(charmask)
                        gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        
                    gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]] #max_gazcharlen：第一句对应的所有词的最大长度  #max_gazlist：第一句BMES包含的单词数的最大值 
                    char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                    gazmask.append(mask)
                    # print(gazmask)#“高”每个label的单词数填充mask
                    #累积 ‘高’B的：[[0, 1]] M:[[0, 1], [0, 1]] E:[[0, 1], [0, 1], [0, 1]] S:[[0, 1], [0, 1], [0, 1], [0, 1]]
                    gazcharmask.append(char_mask)
                    # print(gazcharmask)#“高”每个label的每个单词长度填充mask 
                    #累积 ‘高’B的：[[[0, 0, 1], [1, 1, 1]]] M:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]  E:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]  S:[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]
                layergazmasks.append(gazmask)
                # print(layergazmasks) #“高”所有label的
                #[[[0, 1], [0, 1], [0, 1], [0, 1]]]
                gazchar_masks.append(gazcharmask)
                # print(gazchar_masks)#“高”所有label的
                #[[[[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]]

            texts = ['[CLS]'] + words + ['[SEP]'] #第一句的text
            # print(texts) #第一句  ['[CLS]', '高', '勇', '：', '男', '，', '中', '国', '国', '籍', '，', '无', '境', '外', '居', '留', '权', '，', '[SEP]']
            bert_text_ids = tokenizer.convert_tokens_to_ids(texts)
            # print(bert_text_ids) #第一句 [101, 7770, 1235, 8038, 4511, 8024, 704, 1744, 1744, 5093, 8024, 3187, 1862, 1912, 2233, 4522, 3326, 8024, 102]
            instence_texts.append([words, biwords, chars, gazs, labels])
            # print(instence_texts) #第一句
            # [[words ['高', '勇', '：', '男', '，', '中', '国', '国', '籍', '，', '无', '境', '外', '居', '留', '权', '，'], 
            # biwords ['高勇', '勇：', '：男', '男，', '，中', '中国', '国国', '国籍', '籍，', '，无', '无境', '境外', '外居', '居留', '留权', '权，', '，-null-'], 
            # chars [['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']],
            # gazs 各字对应的BWES单词在gaz_alphabet中的id [[[2, 0], [0, 0], [0, 0], [3, 0]], [[0, 0], [0, 0], [2, 0], [4, 0]],...,[[0, 0], [0, 0], [16, 0], [20, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]], 
            #labels ['B-NAME', 'E-NAME', 'O', 'O', 'O', 'B-CONT', 'M-CONT', 'M-CONT', 'E-CONT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]]
                
            instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids])
            # print(instence_Ids) #第一句
            # [word_Ids [[2, 3, 4, 5, 6, 7, 8, 8, 9, 6, 10, 11, 12, 13, 14, 15, 6],
            #  biword_Ids [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
            # char_Ids [[2], [3], [4], [5], [6], [7], [8], [8], [9], [6], [10], [11], [12], [13], [14], [15], [6]],
            # gaz_Ids 各字的匹配单词(如['高勇'，'高']在gaz_alphabet中的id(如[2, 3])和长度(如[2, 1])  [[[2, 3], [2, 1]], [[4], [1]], [], ..., [3, 2, 1]], [[19], [1]], [[20], [1]], []], 
            # label_Ids [1, 2, 3, 3, 3, 4, 5, 5, 6, 3, 3, 3, 3, 3, 3, 3, 3], 
            # gazs 各字的BMES单词在gaz_alphabet中的id([[[2, 0], [0, 0], [0, 0], [3, 0]], [[0, 0], [0, 0], [2, 0], [4, 0]],...,[[0, 0], [0, 0], [16, 0], [20, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]],
            # gazs_count [[[1, 0], [1, 0], [1, 0], [1, 0]], [[1, 0], [1, 0], [1, 0], [1, 0]], ..., [[1, 0], [1, 0], [185, 0], [1, 0]], [[1, 0], [1, 0], [1, 0], [1, 0]]],
            # gaz_char_Id 各字的BMES单词的字在word_alphabet中的id [[[[2, 3, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[2, 3, 0], [0, 0, 0]], [[3, 0, 0], [0, 0, 0]]], ..., [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[13, 14, 15], [0, 0, 0]], [[15, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]], 
            # layergazmasks 各字的BMES单词数都是两个，第一个是原本的（包含第一次填充的‘NONE’）mask为0，第二个是第二次填充的mask为1 [[[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]], ..., [[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]]], 
            #gazchar_masks 各字的BMES单词的字数都是3个 [[B[[0‘高’, 0‘勇’, 1], [1, 1, 1]], M[[0‘NONE’, 1, 1], [1, 1, 1]], E[[0‘NONE’, 1, 1], [1, 1, 1]], S[[0‘勇’, 1, 1], [1, 1, 1]]], [[[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]], ..., [B[[0, 1, 1], [1, 1, 1]], M[[0, 1, 1], [1, 1, 1]], E[[0'居', 0'留', 0'、权'], [1, 1, 1]], S[[0, 1, 1], [1, 1, 1]]], [[[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]], 
            # bert_text_ids 各字在bert词典中的id [101, 7770, 1235, 8038, 4511, 8024, 704, 1744, 1744, 5093, 8024, 3187, 1862, 1912, 2233, 4522, 3326, 8024, 102]]]

            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids

# 从‘gigaword_chn.all.a2b.uni.ite50’或‘data/ctb.50d.vec’中找到word_alphabet或gaz_alphabet对应的词，将嵌入归一化后存到pretrain_emb中
def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        # print(embedding_path) # "data/gigaword_chn.all.a2b.uni.ite50.vec"  ‘data/ctb.50d.vec’
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
        # print(len(embedd_dict)) # 11327
        # print(embedd_dict['</s>'])
        #[[ 0.008005  0.008839 -0.007661 -0.006556  0.002733  0.006042  0.001882
  #  0.000423 -0.007207  0.004437 -0.008713  0.002499 -0.001503 -0.001914
  # -0.006631 -0.003764  0.005159  0.006051  0.005938  0.003195  0.00309
  # -0.007605 -0.008192  0.009939  0.007603  0.00618  -0.001208  0.008031
  # -0.00099   0.001469 -0.000298 -0.005966  0.002625 -0.002675 -0.007651
  #  0.009508  0.008759 -0.00219  -0.000452  0.001018 -0.007275 -0.008014
  #  0.009109  0.000126 -0.005165 -0.006084 -0.006153  0.003394  0.000403
  #  0.002662]]
        # print(embedd_dim) #50
        
    scale = np.sqrt(3.0 / embedd_dim)
    # print(scale) #0.2449489742783178   0.2449489742783178
    # print(word_alphabet.size())  #1895  12583
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim]) 
    #从一个均匀分布(-scale, scale)中随机采样一个长度为embedd_dim的向量
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict: #从embedd_dict中找到与word_alphabet对应的词，将向量归一化后存入pretrain_emb
            if norm:#true
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1 #完全匹配的
        elif word.lower() in embedd_dict:
            if norm: #true
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1 #小写后匹配的
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1 #不匹配的
    pretrained_size = len(embedd_dict) #词典中的总词数
    # print(len(embedd_dict)) #11327 704368
    # print(len(pretrain_emb)) #word_alphabet gaz_alphabet中的词数 1895 12583
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

