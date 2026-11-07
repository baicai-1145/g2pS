"""Upload weights to HF. Usage: HF_TOKEN=hf_xxx python scripts/push_hf.py"""
import os, sys
from huggingface_hub import HfApi
repo = sys.argv[1] if len(sys.argv) > 1 else 'baicai-1145/g2pS'
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo(repo, repo_type='model', exist_ok=True)
api.upload_large_folder(repo_id=repo, repo_type='model', folder_path='.',
                        ignore_patterns=['.git/*', 'training/*', 'scripts/*', 'examples/*', 'g2ps/*'])
print('uploaded', repo)
