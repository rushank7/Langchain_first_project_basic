import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain ## simplesequentialchain will help to genrate multple outputs
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜 Youtube GPT creator')
prompt = st.text_input('Write your prompt over here')

title_tempalte = PromptTemplate(input_variables=  ['topic'], template = 'write me a youtube video script on Title: {topic}')
script_title_tempalte = PromptTemplate(input_variables=  ['topic'], template = 'write me a youtube video script based on this title  Title: {topic}')


## LLms

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm = llm, prompt = title_tempalte , verbose = True)
script_chain = LLMChain(llm = llm, prompt = script_title_tempalte , verbose = True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain, verbose=True])

# show stuff to screen if there is a prompt
if prompt:
    response = sequential_chain.run(topic=prompt)
    st.write(response)