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
    
    
def make_diff_df(answer_df, predict_df):
    answer_df.rename(columns={'target':'answer_label'}, inplace=True)
    predict_df.rename(columns={'target':'predict_label'}, inplace=True)

    df = pd.concat([answer_df[['ID', 'text', 'answer_label']], predict_df[['predict_label']]], axis=1)

    diff_df = df[df['answer_label']!=df['predict_label']]
    
    labels_dict = {0:'IT과학', 1:'경제', 2:'사회', 3:'생활문화', 4:'세계', 5:'스포츠', 6:'정치'}
    diff_df['answer'] = diff_df['answer_label'].apply(lambda x: labels_dict[x])
    diff_df['predict'] = diff_df['predict_label'].apply(lambda x: labels_dict[x])
    
    return diff_df