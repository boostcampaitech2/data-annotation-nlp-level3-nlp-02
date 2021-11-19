import pickle as pickle
import os
import random
import collections
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class RE_Dataset(Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def split(self, val_ratio) :
        data_size = len(self)
        index_map = collections.defaultdict(list)
        for idx in range(data_size) :
            label = self.labels[idx]
            index_map[label].append(idx)
                
        train_data = []
        val_data = []
            
        label_size = len(index_map)
        for label in range(label_size) :
            idx_list = index_map[label]    
            sample_size = int(len(idx_list) * val_ratio)

            val_index = random.sample(idx_list, sample_size)
            train_index = list(set(idx_list) - set(val_index))
                
            train_data.extend(train_index)
            val_data.extend(val_index)
            
        random.shuffle(train_data)
        random.shuffle(val_data)
            
        train_dset = Subset(self, train_data)
        val_dset = Subset(self, val_data)
        return train_dset, val_dset

def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = [sub['text'] for sub in dataset.subject_entity.map(eval)]
    object_entity = [sub['text'] for sub in dataset.object_entity.map(eval)]
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    return out_dataset

def load_data(dataset):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    if os.path.isfile(dataset) :
        pd_dataset = pd.read_csv(dataset)
    else :
        sheet_url = "https://docs.google.com/spreadsheets/d/1Zfhy3xh-2or5dUsSRm9d3yIN4Jhaleh32Rxqejy8hbI/edit#gid=0"
        pd_dataset = load_googlesheet(sheet_url)
        pd_dataset.to_csv(dataset)
    
    dataset = preprocessing_dataset(pd_dataset)
    return dataset

def load_googlesheet(spreadsheet_url) :
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    #json key file 위치
    json_file_name = '/opt/ml/project/data/theta-webbing-298612-23be65a42b68.json'

    # json key file을 이용하여 접속
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
    gc = gspread.authorize(credentials)

    # 스프레드시트 문서 가져오기
    doc = gc.open_by_url(spreadsheet_url)

    #시트 선택하기
    sheet_name = "final"
    worksheet = doc.worksheet(sheet_name)

    values = worksheet.get_all_values()
    header, rows = values[0], values[1:]

    column_list = ["id","sentence","sentence_with_entity","subject_entity","object_entity","class"]
    data = pd.DataFrame(rows, columns=header)
    data = data[column_list]

    target_data = data[["id", "sentence", "subject_entity", "object_entity", "class"]]
    target_data.rename(columns = {"class" : "label"}, inplace = True)
    return target_data

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    return tokenized_sentences
