import streamlit as st
import torch
import transformers
from dataset import BERTTestDataset
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from model import Model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# BASE_DIR = os.getcwd()
# DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = '2024-level2-datacentric-nlp-8/basic-klue-bert-base'
TOKEN = 'huggingface token'
# MODEL_DIR = os.path.join(BASE_DIR, '../output/checkpoint-300')

def model_setup():
    # print('model setup')
    Tokenizer_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, use_auth_token=TOKEN, num_labels=7)
    # model.parameters

    model.to(DEVICE)
    model.eval()
    return model, tokenizer, DEVICE

def main():
    """
        Datacentric 주제 분류 프로젝트 - 팀 WIZARDS OF SENTENCES -
        팀 동행의 RE(관계 추출) Project의 app.py 코드 참조
    """
    st.markdown("<h2 style='text-align: center; color: red;'>NLP Data-Centric 🦆</h2>", unsafe_allow_html=True)

    st.session_state.category = None
    st.text_input('뉴스제목을 입력하세요', key='sentence')

    def fill_all_inputs():
        if not st.session_state.sentence:
            return False
        return True
    
    if st.button('카테고리 추론'):
        if not fill_all_inputs():
            st.warning('모든 빈칸을 채워주세요')
        else:
            sentence = st.session_state.sentence

            demo_dataset = pd.DataFrame({'text':pd.Series([sentence,])})

            preds = []
            for idx, sample in tqdm(demo_dataset.iterrows()):
                inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                    preds.extend(pred)
                                
            num_to_label_dict = {0: 'IT과학', 1: '경제', 2: '사회', 3: '생활문화', 4: '세계', 5: '스포츠', 6: '정치'}
            answer = num_to_label_dict[preds[0]]
            st.session_state.category = answer

    if st.session_state.category is not None:
        st.write('카테고리 예측 결과 : ', st.session_state.category)


if __name__ == "__main__" :
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer, st.session_state.device = model_setup()
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    device = st.session_state.device
    
    main()
