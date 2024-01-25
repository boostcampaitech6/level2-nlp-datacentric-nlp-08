import os
import random
import numpy as np
import pandas as pd

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

from dataset import BERTDataset
from metrics import compute_metrics

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, './data')
OUTPUT_DIR = os.path.join(BASE_DIR, './output')

def train():    
    model_name = 'klue/bert-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, stratify=data['target'],random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer)
    data_valid = BERTDataset(dataset_valid, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ### for wandb setting
    #os.environ['WANDB_DISABLED'] = 'true'
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

if __name__ == '__main__':
    train()