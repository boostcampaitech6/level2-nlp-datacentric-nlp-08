from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="2024-level2-datacentric-nlp-8/test", 
    filename="kaggle-en-news-500-per-label-exclude-social.csv",
    repo_type="dataset",
    token='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',
    local_dir='./'
    )
