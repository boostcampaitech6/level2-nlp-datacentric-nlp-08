import numpy as np

import evaluate

def compute_metrics(eval_pred):
    f1 = evaluate.load('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')