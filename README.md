# learn_a_field


1. Get an OPEN AI API KEY

2. Download the following files from S3

  
[    Annoy Index for (Arxiv articles upto 12/2022)
](https://arxiv-r-1228.s3.us-west-1.amazonaws.com/annoy_index.ann)    
[Annoy Index for (Arxiv articles since 12/2022 to 08/2023)
](https://arxiv-r-1228.s3.us-west-1.amazonaws.com/annoy_index_since_dec22.ann)   
[Arxiv data with metadata json file (upto 12/2022)
](https://arxiv-r-1228.s3.us-west-1.amazonaws.com/arxiv_12_28_2022.json)   
[Arxiv data with metadata json file (since 12/2022 to 08/2023)
](https://arxiv-r-1228.s3.us-west-1.amazonaws.com/since_dec22.json)

3. pip install -r requirements.txt

4. python lablab_challenge.py

5. You can learn how to build the annoy embeddings from generate_embeddings.py, although you do not need it if you are downloading the index from S3 URLs
