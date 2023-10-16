import json 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time
from annoy import AnnoyIndex

file_name = "arxiv-metadata-oai-snapshot.json"
'''
 Reads the file in path, builds a list of title + abstract
'''
def preprocess(path):
    data = []
    start = time.time()
    print("Begin Loading of Json file:"+path)
    with open(path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    end = time.time()
    ### approximately it takes 50 seconds
    print("Time taken to load json file in memory: ",end-start)

    start = time.time()
    sents = []
    for i in range(len(data)):
        sents.append(data[i]['title']+'[SEP]'+data[i]['abstract'])
    end = time.time()
    print("Time taken to create sentences: ",end-start)  

    return sents

'''Generate embeddings with bath size 400
   change device = CPU if running on CPU
'''
def generate_embeddings(sents,model):
    embeddings = model.encode(sents,batch_size=400,show_progress_bar=True,device='cuda',convert_to_numpy=True)
    np.save("embeddings.npy",embeddings)
    return embeddings


def generate_annoy(embeddings):
    n_trees = 256           
    embedding_size = 768   
    top_k_hits = 5        
    annoy_index = AnnoyIndex(embedding_size, 'angular')
    for i in range(len(embeddings)):
        annoy_index.add_item(i, embeddings[i])
    annoy_index.build(n_trees)
    annoy_index.save("annoy_index.ann")
    return annoy_index

def search(query,annoy_index,embeddings,model):
    query_embedding = model.encode(query,convert_to_numpy=True)
    top_k_hits = 10
    hits = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
    return hits

''' Hardik's code to return the hits'''
def print_results(hits,sents):
    for i in range(len(hits[0])):
        print(sents[hits[0][i]].split('[SEP]')[0])
        print(sents[hits[0][i]].split('[SEP]')[1])
        

sents = preprocess(file_name)
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cuda')
embeddings = generate_embeddings(sents,model)
annoy_index = generate_annoy(embeddings)
query = "cryptography"
hits = search(query,annoy_index,embeddings,model)
print_results(hits,sents)
