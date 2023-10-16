# learn_a_field


Get an OPEN AI API KEY

Download the following files from S3

    Annoy Index for (Arxiv articles upto 12/2022)
    Annoy Index for (Arxiv articles since 12/2022 to 08/2023)
    Arxiv data with metadata json file (upto 12/2022)
    Arxiv data with metadata json file (since 12/2022 to 08/2023)

pip install -r requirements.txt

python lablab_challenge.py

You can learn how to build the annoy embeddings from generate_embeddings.py, although you do not need it if you are downloading the index from S3 URLs
