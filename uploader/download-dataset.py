from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="2024-level2-datacentric-nlp-8/만든레포이름", 
    filename="올릴파일 경로",
    repo_type="dataset",
    token='개인토큰',
    local_dir='내려받을 경로'
    )
