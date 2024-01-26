from huggingface_hub import HfApi
# 1. 터미널에서 huggingface-cli login 명령어를 실행하여 먼저 인증을 거쳐야합니다.
# 2. hugging face 조직(2024-level2-datacentric-nlp-8)에 들어가서 repository를 생성합니다. (private하게)
# 3. 아래 스크립트를 실행해서 모델을 업로드합니다.

# 참고) 모델 불러오기
'''
    from transformers import AutoConfig, AutoModelForSequenceClassification
    
    repo_id = '2024-level2-datacentric-nlp-8/klue-roberta-large-71.01-typed-entity'
    token = 'huggingface token'
    
    model_config = AutoConfig.from_pretrained(repo_id, use_auth_token=token)
    model_config.num_labels = 7
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, config=model_config, use_auth_token=token)
'''

if __name__ == '__main__':
    api = HfApi()

    model_path = '업로드 하려는 모델 경로' # ex '/data/ephemeral/level2-nlp-datacentric-nlp-08/results/checkpoint-9000'
    repo_id = '2024-level2-datacentric-nlp-8/{여러분이 만든 레포이름}' # ex '2024-level2-datacentric-nlp-8/xlm-roberta-large-67-punct'

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model"
    )