from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from config import answer_examples

# 사용할 벡터 인덱스 이름 정의
index_name = 'tax-index'

# 세션별 대화 기록을 저장할 딕셔너리 생성
store = {}


# 세션 ID를 기반으로 대화 히스토리를 가져오거나 새로 생성
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 사전 기반 질문 수정 체인 구성 함수
def get_dictionary_chain():
    dictionary = ['사람을 나타내는 표현 -> 거주자']
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단 된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그럼 경우에는 질문만 리턴해 주세요.
        사전: {dictionary}
        질문: {{question}}
        """
    )

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    
    return llm


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large') # 3072

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    
    retriever = vectorstore.as_retriever(search_kwargs={'k':4})
    
    return retriever
    

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # 대화 히스토리를 고려해 질문을 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # 대화 기반 질문 재구성 프롬프트 템플릿
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 히스토리 인지형 retriever 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever


def get_rag_chain():
    llm = get_llm()
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    history_aware_retriever = get_history_retriever()
    
    # 최종 응답 생성을 위한 QA 시스템 프롬프트
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해 주세요."
        "아래에 제공된 문서를 활용해서 답변해 주시고"
        "답변을 알 수 없다면 모른다고 답변해 주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해 주시고 "
        "2~3 문장 정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    
    # QA 프롬프트 템플릿 정의
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 검색된 문서를 기반으로 응답 생성 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # retriever + QA chain 연결 → RAG 체인 생성
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 세션별로 대화 이력을 관리할 수 있도록 RAG 체인 래핑
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') 

    
    return conversational_rag_chain


# 전체 파이프라인 실행 함수
def get_ai_message(user_message):
    dictionary_chain = get_dictionary_chain()  # 사전 기반 질문 변환
    rag_chain = get_rag_chain()  # RAG 체인 불러오기
    
    # 사전 처리 체인과 RAG 체인을 연결
    tax_chain = {'input': dictionary_chain} | rag_chain 

    # 최종 체인 실행 (세션 ID 지정하여 대화 이력 저장 가능)
    ai_message = tax_chain.stream(
        {
            'question': user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )
    
    return ai_message  # 생성된 응답 스트림 반환
