import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain , SequentialChain ## simplesequentialchain will help to genrate multple outputs
from langchain.prompts import PromptTemplate  ## sequential chain will aloow us to get multiple outputs
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜 Youtube GPT creator')
prompt = st.text_input('Write your prompt over here')

title_tempalte = PromptTemplate(input_variables=  ['topic'], template = 'write me a youtube video script on Title: {topic}')
script_title_tempalte = PromptTemplate(input_variables=  ['topic'], template = 'write me a youtube video script based on this title  Title: {topic}')

## memory
memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')


## LLms

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm = llm, prompt = title_tempalte , verbose = True, output_key='title')
script_chain = LLMChain(llm = llm, prompt = script_title_tempalte , verbose = True, output_key='script')
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],strip_variables=['title','script'] ,verbose=True)

# show stuff to screen if there is a prompt
if prompt:
    response = sequential_chain.run({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message history'):
        st.info(memory.buffer)