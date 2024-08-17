from time import time

import torch
import transformers
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

# Loading the model and tokenizer
model_checkpoints = "meta-llama/Meta-Llama-3.1-8B"
model_config = AutoConfig.from_pretrained(
    model_checkpoints, trust_remote_code=True, max_new_tokens=1024
)

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoints, trust_remote_code=True, config=model_config, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)


# Text generation pipeline and HuggingFacePipeline Object
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_length=3000,
    device_map="auto",
)

llm = HuggingFacePipeline(pipeline=pipeline)


# Verifying by prompting the LLM
# prompt = """<|begin_of_text|>
#            <|start_header_id|>
#              user
#            <|end_header_id|>
#             Dance Monkey
#            <|eot_id|>
#            <|start_header_id|>
#              assistant
#            <|end_header_id|>
#         """
# output = llm.invoke(prompt)
#


def parse(string):
    return string.split("<|end_header_id|")[-1]


# Setting up retriever and database
loader = PyPDFLoader("investment_guidelines.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = GPT4AllEmbeddings()

vectorstore = Chroma(collection_name="sample_collection", embedding_function=embeddings)


vectorstore.add_documents(texts)
retriever = vectorstore.as_retriever(k=7)


class Pipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def retrieve(self, question):
        docs = self.retriever.invoke(question)
        return "\n\n.join([d.page_content for d in docs])"

    def augment(self, question, context):
        return f"""
            <|begin_of_text|>
            <|start_header_id|>
            system
            <|end_header_id|>
                Just give a fooking response
            <|eot_id|>
            <|start_header_id|>
                user
            <|end_header_id|>
                DO NOT LIE MFKER
                Context: {context}
                Question: {question}
            <|eot_id|>
            <|start_header_id|>
                assistant
            <|end_header_id|>"""

    def parse(self, string):
        return string.split("<|end_header_id|>")[-1]

    def generate(self, question):
        context = self.retrieve(question)
        prompt = self.augment(question, context)
        answer = self.llm.invoke(prompt)
        return self.parse(answer)


def llama_chat():
    print("Hello MFKR")
    print("-----------------------------------------------")
    pipe = Pipeline(llm, retriever)
    question = input()

    while question != "STOP":
        out = pipe.generate(question)
        print(out)

        print("Go on")
        print("---------------------------------------------")

        question = input()


llama_chat()
