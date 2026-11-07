"""Upload weights to ModelScope. Usage: MS_TOKEN=xxx python scripts/push_ms.py [namespace/g2pS]"""
import os
import sys

from modelscope.hub.api import HubApi

repo = sys.argv[1] if len(sys.argv) > 1 else 'baicai-1145/g2pS'
api = HubApi()
api.login(os.environ['MS_TOKEN'])
api.create_repo(repo, repo_type='model', exist_ok=True)
# upload weights + meta (skip code dirs)
folders = ['models', 'torch_weights']
files = ['README.md', 'MODEL_CARD.md', 'LICENSE']
for f in files:
    api.upload_file(repo_id=repo, repo_type='model', path_or_fileobj=f, path_in_repo=f)
for d in folders:
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            api.upload_file(repo_id=repo, repo_type='model', path_or_fileobj=p, path_in_repo=p)
print('uploaded', repo)
