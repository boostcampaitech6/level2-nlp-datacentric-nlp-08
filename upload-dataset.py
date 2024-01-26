from datasets import load_dataset
# 1. 터미널에서 huggingface-cli login 명령어를 실행하여 먼저 인증을 거쳐야합니다.
# 2. 허깅페이스의 datasets는 이미 설치되어 있습니다.
# 3. repository를 생성할 필요는 없습니다.
# 3. 아래 스크립트를 실행해서 데이터셋을 업로드합니다.

# 참고) 데이터셋 불러오기
'''
    from datasets import load_dataset
    
    repo_id = '2024-level2-datacentric-nlp-8/train_basic'
    dataset = load_dataset(repo_id,                                 # 허깅페이스에서 불러옴. 로컬에서 불러올때는 load_dataset("csv", data_files="my_file.csv") 이런식으로 씀.
                           revision="main",                         # revision은 데이터셋의 버전인데, 안 넣어도 된다.
                           split="train",                           # split도 안 넣어도 된다.
                           token=True)                              # data_files를 쓰면 한 데이터셋을 train과 validation 데이터셋들로 나눌 수 있음.

    
    
'''

if __name__ == '__main__':

    dataset_path = '업로드 하려는 데이터셋 경로'                         # ex '/data/ephemeral/level2-nlp-datacentric-nlp-08/data/train_editted_by_human.csv'
    dataset_type = "csv"                                               # ex "json"
    repo_id = '2024-level2-datacentric-nlp-8/{만들 레포이름}'           # ex '2024-level2-datacentric-nlp-8/train_repo_name'
    dataset = load_dataset(dataset_type, data_files=dataset_path)      
    dataset.push_to_hub(repo_id=repo_id, private=True)                 # 데이터셋은 parquet 확장자로 업로드된다.