import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(answer_df, predict_df):
    '''예측 결과와 정답 dataframe을 이용해 confusion matrix를 출력합니다.

    Args:
        answer_df (dataframe): 정답 dataframe
        predict_df (dataframe):inference를 한 예측 dataframe
    '''
    y_true = answer_df['target']
    y_pred = predict_df['target']
    cm = confusion_matrix(y_true, y_pred)
    labels = ['ITscience(0)', 'Economy(1)', 'Society(2)', 'LifestyleCulture(3)', 'World(4)', 'Sports(5)', 'Politics(6)']
    sns.heatmap(cm, annot=True, cmap='Blues', fmt = 'd', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()