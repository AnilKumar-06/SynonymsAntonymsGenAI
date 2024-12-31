#from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

chat = chat = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=20,
    timeout=None,
    max_retries=2,
    api_key=api_key,  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

class Commasepratedoutput(BaseOutputParser):
    def parse(self, text:str):
        return text.strip().split(",")
    

st.set_page_config(page_title="Synonyms Generator")
st.header("Synonyms Generator")

template = "You are a helpful assistant. When the user provides any input, you should generate 2 words synonyms in a comma-seprated list."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template)
])

llm_chain = LLMChain(llm = chat, prompt=chat_prompt, output_parser=Commasepratedoutput())
input_text = st.text_input("Enter a word or phrase:", "")

if st.button("Generate Synonyms"):
    if input_text:
        response = llm_chain.predict(text=input_text)
        st.write("Synonyms:", ",".join(response))
    else:
        st.write("Please enter a word or phrase.")