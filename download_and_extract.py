# download_and_extract.py

import os
import tarfile
import urllib.request

def download_and_extract_imdb(data_dir='aclImdb'):
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    file_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    urllib.request.urlretrieve(url, file_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)

if __name__ == '__main__':
    download_and_extract_imdb()
