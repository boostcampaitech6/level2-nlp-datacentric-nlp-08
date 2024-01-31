import os
import random
import numpy as np
import pandas as pd

import torch

from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from dataset import BERTTestDataset

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = os.path.join(BASE_DIR, '../output/checkpoint-300')

def test():  
    model_name = 'klue/bert-base'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    tokenized_dataset = BERTTestDataset(dataset_test, tokenizer)
    
    batch_size = 19  # 테스트 데이터셋 크기가 47785개 = 5*19*503
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    for idx, sample_batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            outputs = model(input_ids=sample_batch['input_ids'].to(DEVICE), attention_mask=sample_batch['attention_mask'].to(DEVICE))
            logits_batch = outputs.logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits_batch), dim=1).cpu().numpy()
            preds.extend(pred)
            
    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join(BASE_DIR, 'output.csv'), index=False)
                
if __name__ == '__main__':
    test()