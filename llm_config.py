
from  langchain_groq import ChatGroq


api_key = "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"

def get_model():
    llm = ChatGroq(

    groq_api_key=api_key,
    model='llama-3.3-70b-versatile'
 )
    return llm




