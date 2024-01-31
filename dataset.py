import torch

from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []
        self.labels = []
        
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)
    
class BERTTestDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        # self.inputs = []
        
        self.tokenized_inputs = tokenizer(input_texts.tolist(), padding=True, return_tensors='pt')
        # for text in input_texts:
        #     tokenized_input = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        #     self.inputs.append(tokenized_input)
            
    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_inputs['input_ids'][idx].squeeze(0),  
            'attention_mask': self.tokenized_inputs['attention_mask'][idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])