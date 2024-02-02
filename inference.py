import os
import random
import numpy as np
import pandas as pd

import torch

from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, './data')
MODEL_DIR = os.path.join(BASE_DIR, './output/checkpoint-2100')

def test():  
    model_name = 'klue/bert-base'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    model.eval()
    preds = []
    for idx, sample in tqdm(dataset_test.iterrows()):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
            
    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join(BASE_DIR, 'output.csv'), index=False)
                
if __name__ == '__main__':
    test()