from gpt_index import SimpleDirectoryReader, ServiceContext, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import os

"""
Python script to train ChatGPT on docs folder.
Copied from https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api/
and added fix from comments.
secret_key.txt and docs folder are in the .gitignore.
"""
with open('secret_key.txt', 'r') as file:
    api_key = file.read()

os.environ["OPENAI_API_KEY"] = api_key


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context)
    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(
                         lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)
