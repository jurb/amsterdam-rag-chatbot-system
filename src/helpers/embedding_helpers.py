from openai import AzureOpenAI, OpenAI
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAIEmbeddingFunction:
    def __init__(self, model: str = None):
        # Determine if we should use Azure OpenAI
        if os.getenv("AZURE_API_KEY") and os.getenv("AZURE_EMB_DEPLOYMENT"):
            # Use Azure OpenAI
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=os.getenv("AZURE_EMB_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_API_ENDPOINT")
            )
            # Use the Azure deployment name for embeddings
            self.model = os.getenv("AZURE_EMB_DEPLOYMENT")
            self.is_azure = True
        else:
            # Fall back to standard OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            # Use the provided model or default to text-embedding-3-large
            self.model = model or "text-embedding-3-large"
            self.is_azure = False
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model=self.model)
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding