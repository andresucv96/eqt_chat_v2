from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone as pine
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
import chainlit as cl
import config

text_field = "content"  # the metadata field that contains our text


def init_vectorstore():
    # initialize the vector store object
    pc = pine(api_key=config.PINECONE_API_KEY)
    index = pc.Index("diegoindex")
    embed_model = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    api_key=config.OPENAI_API_KEY
                )

    vectorstore = PineconeVectorStore(
                index=index,
                embedding=embed_model,
                text_key=text_field
                )

    return vectorstore


# set memory object with memory variables
memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                    )

memory.load_memory_variables({})
chat_memory = ChatMessageHistory()

# create vectorstore object
vectorstore = init_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True,
                       model=config.OPENAI_COMPLETION_MODEL,
                       openai_api_key=config.OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are a helpful chatbot that answer questions about the following context {context}"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | model
        | output_parser
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_memory,
        input_messages_key="question",
        history_messages_key="history",
    )

    cl.user_session.set("runnable", chain_with_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config={
            'configurable': {'session_id': 'chat_memory'},
            'callbacks': [cl.LangchainCallbackHandler()]
            }
    ):
        await msg.stream_token(chunk)

    await msg.send()
