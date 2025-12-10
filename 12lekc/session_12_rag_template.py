from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import ollama
from torch.hub import download_url_to_file

model_embeddings = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True,
    return_sparse=False,
    return_colbert_vecs=False, #
    return_dense=True
)

def path_workaround():
    ## why this bug hapened!?
    curdir = os.getcwd()
    print(f"curdir {curdir}")
    root = 'data'
    # if not curdir.endswith('PyCharmMiscProject'):
    #    rootpath = '../' + root
    path_dataset = f'{root}/datasets'

    print(f"{path_dataset} exists: {os.path.exists(path_dataset)}")
    return root, path_dataset


root, dataset_path = path_workaround()

os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
path_dataset = dataset_path+"/movies_info.csv"
if not os.path.exists(path_dataset):
    # https://www.kaggle.com/datasets/rushildhingra25/movies-info?select=movies_info.csv
    download_url_to_file(
        "https://share.yellowrobot.xyz/quick/2025-8-1-F9927D15-6D09-44AD-89F5-169A01A8C6CF.csv",
        path_dataset,
        progress=True
    )
df_movies = pd.read_csv(path_dataset)
# columns: original_title, overview, genres (["a", "b"]) 

embs_dense_overviews = []
path_embeddings = dataset_path+"/movies_embeddings.pkl"
if not os.path.exists(path_embeddings):
    download_url_to_file(
        "https://share.yellowrobot.xyz/quick/2025-12-10-1B3250CF-D5B8-4E9B-969B-098832E60DB7.pkl",
        path_embeddings,
        progress=True
    )
if os.path.exists(path_embeddings):
    with open(path_embeddings, "rb") as f:
        embs_dense_overviews = pickle.load(f)
else:
    batch_size = 10
    overview_texts = df_movies["overview"].values
    idx = 0
    for overview_text in tqdm(overview_texts, desc="Encoding movie overviews"):
        print(idx, overview_text)
        idx += 1
        dense_vecs = np.zeros(1024, dtype=np.float16)
        embedding = model_embeddings.encode(overview_text)
        if embedding['dense_vecs'] is not None:
            dense_vecs = embedding['dense_vecs']
        embs_dense_overviews.append(dense_vecs)
    with open(path_embeddings, "wb") as f:
        pickle.dump(embs_dense_overviews, f)
embs_dense_overviews = np.array(embs_dense_overviews)
    
# TODO
print("OK")