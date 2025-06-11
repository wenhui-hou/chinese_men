import pandas as pd
from gensim.summarization import bm25
import pickle
from random import shuffle

entities = []
entities_cut = []
with open('./data/CHIP-CDN/CHIP-CDN_train.json', 'r') as f:
    train_list = json.load(f)
for i in range(len(train_list)):
    train_list[i]['text'] = _process_single_sentence(train_list[i]['text'], mode="text")
    train_list[i]['normalized_result'] = _process_single_sentence(train_list[i]['normalized_result'], mode='normalized_result')
train_df=pd.DataFrame(train_list)
train_data = train_df.values
train_datas = {}
train_mentions = []
train_mentions_cut = []
for i in range(len(train_data)):
    train_datas[train_data[i][0]] = train_data[i][1].split("##")
    for gold in train_data[i][1].split("##"):
        if gold not in entities:
            entities.append(gold)
            entities_cut.append(list(gold))
    train_mentions.append(train_data[i][0])
    train_mentions_cut.append(list(train_data[i][0]))

with open('./data/CHIP-CDN/CHIP-CDN_dev.json', 'r') as f:
    val_list = json.load(f)
for i in range(len(val_list)):
    val_list[i]['text'] = _process_single_sentence(val_list[i]['text'], mode="text")
    val_list[i]['normalized_result'] = _process_single_sentence(val_list[i]['normalized_result'], mode='normalized_result')
val_df=pd.DataFrame(val_list)
val_data = val_df.values
val_datas = {}
val_mentions = []
val_mentions_cut = []
for i in range(len(val_data)):
    val_datas[val_data[i][0]] = val_data[i][1].split("##")
    for gold in val_data[i][1].split("##"):
        if gold not in entities:
            entities.append(gold)
            entities_cut.append(list(gold))
    val_mentions.append(val_data[i][0])
    val_mentions_cut.append(list(val_data[i][0]))

with open('./data/CHIP-CDN/CHIP-CDN_test.json', 'r') as f:
    answer_list = json.load(f)
for i in range(len(answer_list)):
    answer_list[i]['text'] = _process_single_sentence(answer_list[i]['text'], mode="text")
    answer_list[i]['normalized_result'] = _process_single_sentence(answer_list[i]['normalized_result'], mode='normalized_result')
answer_df=pd.DataFrame(answer_list)
answer_data = answer_df.values

pickle.dump(train_datas, open('./data/CHIP-CDN/train_mentions_gold_CHIP-CDN', 'wb'), -1)
pickle.dump(val_datas, open('./data/CHIP-CDN/test_mentions_gold_CHIP-CDN', 'wb'), -1)