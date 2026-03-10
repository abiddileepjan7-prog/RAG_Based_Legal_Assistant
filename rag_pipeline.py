import os
import torch
from threading import Thread

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

from langchain_core.language_models.llms import LLM
from typing import Iterator, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = 'nf4'
)

# from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
# from threading import Thread
print("Loading Mistral model...")
model_id = 'mistralai/Mistral-7B-Instruct-v0.2'

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    quantization_config = bnb_config
)

def create_streaming_llm():

    def stream_generate(prompt):

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

    return stream_generate

from langchain_core.language_models.llms import LLM
from typing import Iterator, List, Optional

class StreamingLLM(LLM):

  @property
  def _llm_type(self) -> str:
    return 'streaming_mistral'

  def _call(self, prompt: str, stop :  Optional[List[str]] = None) -> str:
    return "".join(self.stream(prompt))

  def stream(self, prompt: str) -> Iterator[str]:
    generator = create_streaming_llm()
    for token in generator(prompt):
      yield token

llm = StreamingLLM()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os

folder = 'files'


documents = []

if not os.path.exists(folder):
    raise ValueError(f"Folder '{folder}' not found")

print("Loading documents...")

for file in os.listdir(folder):
    if file.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(folder, file))
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

vectorstore = FAISS.from_documents(docs, embeddings)

print("Building vector store...")

retriever = vectorstore.as_retriever(search_kwargs={'k':5})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

reranker_model = HuggingFaceCrossEncoder(
    model_name='BAAI/bge-reranker-base'
)

reranker = CrossEncoderReranker(
    model=reranker_model,
    top_n=2
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=multi_retriever,
    base_compressor=reranker
)
def add_documents(file_paths):

    global vectorstore, retriever, multi_retriever, compression_retriever

    new_docs = []

    for path in file_paths:

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()

            # ⭐ Skip empty pages
            loaded = [d for d in loaded if d.page_content.strip() != ""]

            new_docs.extend(loaded)

    if len(new_docs) == 0:
        print("⚠️ No valid documents found")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(new_docs)

    # ⭐ Remove empty chunks again
    split_docs = [d for d in split_docs if d.page_content.strip() != ""]

    if len(split_docs) == 0:
        print("⚠️ No valid chunks after splitting")
        return

    # ⭐ Add safely
    vectorstore.add_documents(split_docs)

    # ⭐ Rebuild retriever chain
    retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multi_retriever,
        base_compressor=reranker
    )

    print("✅ New documents added to vector store")


from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key = 'chat_history',
    return_messages=True
)

from langchain.prompts import PromptTemplate

system_prompt = """
You are an intelligent AI assistant with access to external documents.

You have two modes:

MODE 1 — Document Mode (RAG)
If relevant information is found in the provided context:
- Use the context as the primary source
- Base your answer strictly on the documents
- Do not contradict the documents
- Mention limitations if information is incomplete

MODE 2 — General Knowledge Mode
If the context does NOT contain useful information:
- Answer using your own knowledge like a normal AI assistant
- Be helpful, accurate, and conversational

Guidelines:
- Prefer document information when available
- Do not mention "context" explicitly to the user
- Do not say "based on my training data"
- Keep responses natural and clear


Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer:
"""



prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "question", "chat_history"]
)

def generate_answer_stream(question):

    docs = compression_retriever.invoke(question)
    print("Retrieved docs:", len(docs))

    context = "\n\n".join([d.page_content for d in docs]) if docs else "NO_RELEVANT_DOCUMENTS"

    history_messages = memory.load_memory_variables({})["chat_history"]

    history = "\n".join([
        f"{msg.type}: {msg.content}"
        for msg in history_messages
    ])

    final_prompt = prompt.format(
        context=context,
        question=question,
        chat_history=history
    )

    answer_text = ""

    for token in llm.stream(final_prompt):
        answer_text += token
        yield token

    memory.save_context(
        {"input": question},
        {"output": answer_text}
    )

print("RAG pipeline ready ✅")