# -*- coding: utf-8 -*-

import sys
import numpy as np
from creat_lattices.alphabet import Alphabet
from creat_lattices.functions import *
from creat_lattices.gazetteer import Gazetteer
import pickle
import pandas as pd

import json
import pandas as pd
def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def _process_single_sentence(sentence, mode='text'):
    sentence = str_q2b(sentence)
    sentence = sentence.strip('"')
    if mode == "text":
        sentence = sentence.replace("\\", ";")
        sentence = sentence.replace(",", ";")
        sentence = sentence.replace("、", ";")
        sentence = sentence.replace("?", ";")
        sentence = sentence.replace(":", ";")
        sentence = sentence.replace(".", ";")
        sentence = sentence.replace("/", ";")
        sentence = sentence.replace("~", "-")
    return sentence


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    global in_lines, line,pairs,word,label,biword, char,seqlen
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.gaz_count = {}
        self.gaz_split = {}
        self.biword_count = {}

        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True
        self.HP_use_count = False

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 128
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

        self.HP_num_layer = 4

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Biword alphabet size: %s"%(self.biword_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Biword embedding size: %s"%(self.biword_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))


    def build_alphabet(self, input_file):
        global in_lines, line,pairs,word,label,biword, char,seqlen
        print(input_file) 
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        # print(len(in_lines)) #list 127920
        seqlen = 0        
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            # print(line)  #str 高 B-NAME
            # print(len(line)) #9
            if len(line) > 2:
                pairs = line.strip().split()
                # print(pairs) #['高', 'B-NAME']
                word = pairs[0]  
                if self.number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) #高
                label = pairs[-1]
                self.label_alphabet.add(label) #self.label_alphabet = Alphabet('label', True)
                self.word_alphabet.add(word)  #self.word_alphabet = Alphabet('word')
                if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2: 
                    biword = word + in_lines[idx+1].strip().split()[0] 
                else:  
                    biword = word + NULLKEY
                # print(biword) 
                self.biword_alphabet.add(biword) #self.biword_alphabet = Alphabet('biword')
                #self.biword_count = {}
                self.biword_count[biword] = self.biword_count.get(biword,0) + 1  
                for char in word:
                    # print(char) 
                    self.char_alphabet.add(char) #self.char_alphabet = Alphabet('character')
                seqlen += 1
            else:
                seqlen = 0
        # print(seqlen) #0        
        
        self.word_alphabet_size = self.word_alphabet.size()
        # print(self.word_alphabet_size) #train 1785  #dev 1838  #test 1895  
        #dev=train+[1786-1838]; test=dev+[1839,1895]
        self.biword_alphabet_size = self.biword_alphabet.size()
        # print(self.biword_alphabet_size) #18605 #20043 #21408
        self.char_alphabet_size = self.char_alphabet.size()
        # print(self.char_alphabet_size) #1785  #1838  #1895
        self.label_alphabet_size = self.label_alphabet.size()
        # print(self.label_alphabet_size) #29  #29  #29
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"  
            else:
                self.tagScheme = "BIO"
        # print(self.tagScheme) #BMES  #BMES  #BMES
        
    def build_alphabet_yidu(self, input_file):
        global in_lines, line,pairs,word,label,biword, char,seqlen
        print(input_file) 
#         in_lines= pickle.load(open(input_file, 'rb')) 
        
        in_lines= pd.read_excel(input_file) 
                
        in_lines= in_lines.to_dict(orient='records')
        in_lines= {item['原始词']: item['标准词'] for item in in_lines}
#         in_lines= {item['text']: item['normalized_result'] for item in in_lines}
        
        # print(len(in_lines)) #dict
        seqlen = 0   
        for item in in_lines.items():
            key = item[0]
            value = item[1]
            # print(type(key))
            # print(type(value))
            for word in key:
                # print(word)
                if self.number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) 
                self.word_alphabet.add(word)
                for char in word:
                    # print(char)
                    self.char_alphabet.add(char) 
    
            for words in value:
                for word in words:
                    # print(word)
                    if self.number_normalized:  #True
                        word = normalize_word(word)
                        # print(word)
                    self.word_alphabet.add(word)          
                    for char in word:
                    # print(char) 
                        self.char_alphabet.add(char) 
        
                seqlen += 1
            else:
                seqlen = 0
        # print(seqlen) #0        
        
        self.word_alphabet_size = self.word_alphabet.size()
        print(self.word_alphabet_size) #train 1785  #dev 1838  #test 1895  
        #dev=train+[1786-1838]; test=dev+[1839,1895]
        self.char_alphabet_size = self.char_alphabet.size()
        print(self.char_alphabet_size) #1785  #1838  #1895

        self.tagScheme = "BMES"  
        
    def build_alphabet_CDN(self, input_file):
        global in_lines, line,pairs,word,label,biword, char,seqlen
        print(input_file) 
#         in_lines= pickle.load(open(input_file, 'rb')) 
        
#         in_lines= pd.read_excel(input_file) 
        
        import json
        with open(input_file, 'r') as f: 
            in_lines = json.load(f)  
        for i in range(len(in_lines)):
            in_lines[i]['text'] = _process_single_sentence(in_lines[i]['text'], mode="text")
            in_lines[i]['normalized_result'] = _process_single_sentence(in_lines[i]['normalized_result'], mode='normalized_result')
        in_lines=pd.DataFrame(in_lines)
                
        in_lines= in_lines.to_dict(orient='records')

        in_lines= {item['text']: item['normalized_result'] for item in in_lines}
        
        # print(len(in_lines)) #dict
        seqlen = 0   
        for item in in_lines.items():
            key = item[0]
            value = item[1]
            # print(type(key))
            # print(type(value))
            for word in key:
                # print(word)
                if self.number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) 
                self.word_alphabet.add(word)
                for char in word:
                    # print(char) 
                    self.char_alphabet.add(char) 
    
            for words in value:
                for word in words:
                    # print(word)
                    if self.number_normalized:  #True
                        word = normalize_word(word)
                        # print(word) 
                    self.word_alphabet.add(word)          
                    for char in word:
                    # print(char)
                        self.char_alphabet.add(char) 
        
                seqlen += 1
            else:
                seqlen = 0
        # print(seqlen) #0        
       
     
        
        self.word_alphabet_size = self.word_alphabet.size()
        print(self.word_alphabet_size) #train 1785  #dev 1838  #test 1895  
        #dev=train+[1786-1838]; test=dev+[1839,1895]
        self.char_alphabet_size = self.char_alphabet.size()
        print(self.char_alphabet_size) #1785  #1838  #1895

        self.tagScheme = "BMES" 


    def build_gaz_file(self, gaz_file):
        ## build gaz file,initial read gaz embedding file
        if gaz_file:
            # print(gaz_file) 
            fins = open(gaz_file, 'r',encoding="utf-8").readlines()

            for fin in fins:
                fin = fin.strip().split()[0]                
                if fin:
                    #from utils.gazetteer import Gazetteer
                    self.gaz.insert(fin, "one_source") #self.gaz = Gazetteer(self.gaz_lower)         
            
            print ("Load gaz file: ", gaz_file, " total size:", self.gaz.size())  #704368
        else:
            print ("Gaz file is None, load nothing")


    def build_gaz_alphabet(self, input_file, count=False):
        global word_list,w_length,matched_entity,longest,longest_index,gazlen
        print(input_file)  
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else: 
                w_length = len(word_list) 
                entitys = []
                
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    entitys += matched_entity                   
                    
                    for entity in matched_entity:
                        self.gaz_alphabet.add(entity) #self.gaz_alphabet = Alphabet('gaz')
                        index = self.gaz_alphabet.get_index(entity) 
                        #self.gaz_count = {}
                        self.gaz_count[index] = self.gaz_count.get(index,0)  ## initialize gaz count
                
                
                # print(count) #True
                if count:  #
                    entitys.sort(key=lambda x:-len(x))                    
                    while entitys:
                        longest = entitys[0]
                        longest_index = self.gaz_alphabet.get_index(longest)
                        # print(longest_index) #16 2  6... 
                        self.gaz_count[longest_index] = self.gaz_count.get(longest_index, 0) + 1
                        # print(self.gaz_count[longest_index])#1 1 1... 
                        gazlen = len(longest)
                        # print(gazlen) #3 2 2...
                        for i in range(gazlen):
                            for j in range(i+1,gazlen+1):
                                covering_gaz = longest[i:j]
                                if covering_gaz in entitys:
                                    entitys.remove(covering_gaz)
                                    # print('remove:',covering_gaz)
                word_list = []
                
        
        print("gaz alphabet size:", self.gaz_alphabet.size())  #train 11224 #dev 11909 #test 12583
        
    def build_gaz_alphabet_yidu(self, input_file, count=False):
        global word_list,w_length,matched_entity,longest,longest_index,gazlen
        print(input_file) 
#         in_lines= pickle.load(open(input_file, 'rb'))
        
        in_lines= pd.read_excel(input_file) 
        

        in_lines= in_lines.to_dict(orient='records')
        in_lines= {item['原始词']: item['标准词'] for item in in_lines}        
#         in_lines= {item['text']: item['normalized_result'] for item in in_lines}        
                        
        # print(len(in_lines)) #dict
        word_list = []
        for item in in_lines.items():
            key = item[0]
            # print(key)
            value = item[1]
            for word in key:
                # print(word)
                if self.number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) 
                word_list.append(word)
            # print(word_list)

            for words in value:
                for word in words:
                    # print(word)
                    if self.number_normalized:  #True
                        word = normalize_word(word)
                        # print(word) 
                    word_list.append(word)
                    # print(word_list)
        
            # print(len(word_list)) #157 

            w_length = len(word_list) 
            entitys = []
              
            for idx in range(w_length):
                matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                entitys += matched_entity                   
                    
                for entity in matched_entity:
                    self.gaz_alphabet.add(entity) #self.gaz_alphabet = Alphabet('gaz')
                    index = self.gaz_alphabet.get_index(entity) 
                    #self.gaz_count = {}
                    self.gaz_count[index] = self.gaz_count.get(index,0)  ## initialize gaz count
                
            if count:  #
                entitys.sort(key=lambda x:-len(x))                    
                while entitys:
                    longest = entitys[0]
                    longest_index = self.gaz_alphabet.get_index(longest)
                    # print(longest_index) #16 2  6... 
                    self.gaz_count[longest_index] = self.gaz_count.get(longest_index, 0) + 1
                    # print(self.gaz_count[longest_index])#1 1 1... 
                    gazlen = len(longest)
                    # print(gazlen) #3 2 2...
                    for i in range(gazlen):
                        for j in range(i+1,gazlen+1):
                            covering_gaz = longest[i:j]
                            if covering_gaz in entitys:
                                entitys.remove(covering_gaz)
                                # print('remove:',covering_gaz)
            word_list = []
        
        print("gaz alphabet size:", self.gaz_alphabet.size())
        
        
    def build_gaz_alphabet_CDN(self, input_file, count=False):
        global word_list,w_length,matched_entity,longest,longest_index,gazlen
        print(input_file) 
#         in_lines= pickle.load(open(input_file, 'rb'))
        
#         in_lines= pd.read_excel(input_file)  #
        
        import json
        with open(input_file, 'r') as f: 
            in_lines = json.load(f)  
        for i in range(len(in_lines)):
            in_lines[i]['text'] = _process_single_sentence(in_lines[i]['text'], mode="text")
            in_lines[i]['normalized_result'] = _process_single_sentence(in_lines[i]['normalized_result'], mode='normalized_result')
        in_lines=pd.DataFrame(in_lines)
                       
        in_lines= in_lines.to_dict(orient='records')
        in_lines= {item['text']: item['normalized_result'] for item in in_lines}        
                        
        # print(len(in_lines)) #dict
        word_list = []
        for item in in_lines.items():
            key = item[0]
            # print(key)
            value = item[1]
            for word in key:
                # print(word)
                if self.number_normalized:  #True
                    word = normalize_word(word)
                    # print(word) 
                word_list.append(word)
            # print(word_list)

            for words in value:
                for word in words:
                    # print(word)
                    if self.number_normalized:  #True
                        word = normalize_word(word)
                        # print(word) 
                    word_list.append(word)
                    # print(word_list)
        
            # print(len(word_list)) #157 

            w_length = len(word_list) 
            entitys = []
              
            for idx in range(w_length):
                matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                entitys += matched_entity                   
                    
                for entity in matched_entity:
                    self.gaz_alphabet.add(entity) #self.gaz_alphabet = Alphabet('gaz')
                    index = self.gaz_alphabet.get_index(entity) 
                    #self.gaz_count = {}
                    self.gaz_count[index] = self.gaz_count.get(index,0)  ## initialize gaz count
                
            
            if count:  #
                entitys.sort(key=lambda x:-len(x))                    
                while entitys:
                    longest = entitys[0]
                    longest_index = self.gaz_alphabet.get_index(longest)
                    # print(longest_index) #16 2  6... 
                    self.gaz_count[longest_index] = self.gaz_count.get(longest_index, 0) + 1
                    # print(self.gaz_count[longest_index])#1 1 1... 
                    gazlen = len(longest)
                    # print(gazlen) #3 2 2...
                    for i in range(gazlen):
                        for j in range(i+1,gazlen+1):
                            covering_gaz = longest[i:j]
                            if covering_gaz in entitys:
                                entitys.remove(covering_gaz)
                                # print('remove:',covering_gaz)
            word_list = []
           
        
        print("gaz alphabet size:", self.gaz_alphabet.size())           

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print ("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print ("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)


    def build_gaz_pretrain_emb(self, emb_path):
        print ("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,  self.gaz_emb_dim, self.norm_gaz_emb)


    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz,self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet,self.biword_alphabet, self.biword_count, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





