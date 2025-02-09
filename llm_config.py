
from  langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

def get_model():
  llm = ChatGroq(groq_api_key=api_key, model = 'gemma2-9b-it')
  return llm