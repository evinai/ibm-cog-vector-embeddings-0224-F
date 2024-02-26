examples = [
  "Dogs are a man's best friend.",
  "There are no animals with faster reflexes than cats.",
  "The global pet birth rate has been increasing since 2008.",
  "Humans and dogs are very good friends.",
  "Some pets can be more loyal than others.",
  "Felines are very quick to react to sudden events."
]

import requests
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUG_API")


model_id = "sentence-transformers/all-MiniLM-L6-v2"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


response = requests.post(api_url, headers=headers, json={"inputs": examples, "options":{"wait_for_model":True}})
embeddings = response.json()

print(embeddings)

print("Number of lists returned:", len(embeddings))
print("Length of each list:", len(embeddings))


def dot_product(A, B):
    return sum(i[0] * i[1] for i in zip(A, B))


import pandas as pd

# Create a dict with the example data
data = {
    'text': examples,
    'vector': embeddings
}

# Create a DataFrame using the dictionary
df = pd.DataFrame.from_dict(data)

user_article_headline = 'Why humans get along so well with dogs'

# Fetch the vector embedding of the headline of the article that the user is currently reading
response = requests.post(api_url, headers=headers, json={"inputs": [user_article_headline], "options":{"wait_for_model":True}})
user_article_embedding = response.json()[0]

# Get list of vectors from our database table
vectors = df["vector"].values.tolist()

# Create a list of the dot products of user_article_embedding against each vector in the database
similarities = [dot_product(user_article_embedding, vector) for vector in vectors]
print("Similarity scores:", similarities)

# Get the index of the value with the highest similarity score
max_similarity_index = similarities.index(max(similarities))

# Fetch the sentence corresponding to that index from the dataframe
sentences = df["text"].values.tolist()
most_similar_sentence = sentences[max_similarity_index]

print(most_similar_sentence)

'''
Similarity scores: [0.6367540860345434, 0.25669272034699475, 0.3402974996977122, 0.7539815812800277, 0.6026046867479701, 0.33806413301719257]

Humans and dogs are very good friends.

'''

# Convert only the embeddings to a DataFrame, without the texts
df_embeddings = pd.DataFrame(embeddings)

# Export the dataframe as a .csv file
df_embeddings.to_csv("embeddings.csv", index=False)

# After uploading our dataset, we can load the embedded dataset from Hugging Face using the datasets library and convert
# it to a PyTorch FloatTensor, which is one way to operate on the data. Make sure to replace namespace/repo_name with your
# user and repo name.
import torch
from datasets import load_dataset

articles_embeddings = load_dataset("tog-ai/my_test_dataset")
dataset_embeddings = torch.from_numpy(articles_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

cat_article = ['Cats have very fast reflexes']
response = requests.post(api_url, headers=headers, json={"inputs": cat_article, "options":{"wait_for_model":True}})

query_embeddings = torch.FloatTensor(response.json())

#Given any article name, suppose we would like to conduct semantic search to generate relevant suggestions. 
# Similar to the previous section, we first get the embeddings for that article.Given any article name, suppose we 
# would like to conduct semantic search to generate relevant suggestions. Similar to the previous section, we first 
# get the embeddings for that article.
cat_article = ['Cats have very fast reflexes']
response = requests.post(api_url, headers=headers, json={"inputs": cat_article, "options":{"wait_for_model":True}})

query_embeddings = torch.FloatTensor(response.json())


# Next, we can use the sentence_transformers library to query our data. Let’s use it to find the most relevant 2 articles.
from sentence_transformers.util import semantic_search

# Find top 2 similar vectors using semantic_search
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=2)

# Print result
print(hits)


# The corpus_id field in each row refers to the index at which the ‘similar’ vector was found.
# We can use this value to index the text from our starting data.

print([examples[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])