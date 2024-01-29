from huggingface_hub import HfApi
# 1. 터미널에서 huggingface-cli login 명령어를 실행하여 먼저 인증을 거쳐야합니다.
# 2. hugging face 조직(2024-level2-re-nlp-8)에 들어가서 repository를 생성합니다. 
# 3. 아래 스크립트를 실행해서 csv파일을 업로드합니다.

if __name__ == '__main__':
    api = HfApi()

    data_path = '{올리고 싶은 데이터셋 경로}' # ex '/data/ephemeral/level2-nlp-datacentric-nlp-08/results/checkpoint-9000'
    repo_id = '2024-level2-datacentric-nlp-8/{생성한 레포 이름}' # ex '2024-level2-datacentric-nlp-8/xlm-roberta-large-67-punct'

    api.upload_folder(
        folder_path=data_path,
        repo_id=repo_id,
        repo_type="dataset"
    )