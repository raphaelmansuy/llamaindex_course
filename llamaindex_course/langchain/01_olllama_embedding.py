""" Example of using the OllamaEmbeddings class to embed a query. """

from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings

TEXT_TO_ENCODE = "Dog"

TEXT_TO_COMPARE = "Wolf"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

query_result = embeddings.embed_query(TEXT_TO_ENCODE)

# print the first 5 results
print(query_result[:5])

# print the full vector
print(f"Vector length: {len(query_result)}")
print(query_result)

# compare the two texts

# embed the second text
query_result_2 = embeddings.embed_query(TEXT_TO_COMPARE)

# calculate the cosine similarity
# 1 means the same, 0 means different

similarity = cosine_similarity([query_result], [query_result_2])

print(f"Similarity: {similarity[0][0]}")
