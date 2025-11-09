'''

'''
from google.cloud.firestore_v1.vector import Vector
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv


'''
Based on https://ai.google.dev/gemini-api/docs/embeddings
'''

class EmbeddingMan:
    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def calcEmbedding(self, doc):
        texts = str(doc)
        print(f"Embedding {texts}")
        embedding_config = types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=2048
                    );
        result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=embedding_config
                    ).embeddings
        print(f"Embedding vector: {result}")
        return Vector(result[0].values)
