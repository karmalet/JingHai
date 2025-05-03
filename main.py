import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_teddynote import logging
from dotenv import load_dotenv
from functools import partial
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
# logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("안녕하세요? RAG 기반 QA 봇입니다💬")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    
    # 모드 선택
    mode = st.radio("AI가 참고할 데이터 선택", ("MODE1: 海派京派 소설","MODE2: 업로드한 파일"))
    
    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "MODE 1: LLM 선택", ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o", \
                          "claude-3-5-haiku-latest","claude-3-5-sonnet-latest","claude-3-opus-latest"], index=0
    )

    # 파일 업로드
    uploaded_file = st.file_uploader("MODE2: 여기에 업로드한 파일만 보고 대답합니다 (PDF만가능)", type=["pdf"])

    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def batch_embed_and_store(documents, embeddings, batch_size=200):
    """Split documents into batches and create FAISS vectorstore safely."""
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
    return vectorstore


# 파일이 업로드 되었을 때: 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file=None):
    if file:
        # 사용자가 업로드한 파일이 있다면, 업로드한 파일을 캐시 디렉토리에 저장합니다.
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        # 단계 1: 문서 로드(Load Documents)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        print('로드된 docs 목록')
        for doc in docs:
            print(doc.metadata['source'])
    
        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        global split_documents
        split_documents = text_splitter.split_documents(docs)
        
        # 쪼갠 문서 확인 (디버깅용)
        print(f"Split된 청크 개수: {len(split_documents)}")
        print(f"첫 번째 청크 길이(문자 수): {len(split_documents[0].page_content)}")
    
        # 단계 3: 임베딩(Embedding) 생성
        new_embeddings = OpenAIEmbeddings()
    
    
        # 단계 4: DB 생성(Create DB) 및 저장
        # 벡터스토어를 생성합니다.
        # vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore = batch_embed_and_store(split_documents, new_embeddings, batch_size=200)
        retriever = vectorstore.as_retriever()
        return retriever
    else:
        return None

@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def load_embeddings():
    # # TextLoader를 encoding='utf8'로 고정하는 래퍼 생성
    # CustomTextLoader = partial(TextLoader, encoding='utf8')
    # # 단계 1: 문서 로드(Load Documents)
    # loader = DirectoryLoader("literature", glob="**/*.txt", \
    #                          loader_cls=CustomTextLoader, show_progress=True, use_multithreading=True)

    # 단계 3: 임베딩(Embedding) 생성
    # CacheBackedEmbeddings 저장하고, 또한 저장되어 있는 임베딩 가져오기
    store = LocalFileStore("./.cache/embeddings") # cache 저장 경로 지정
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace=embeddings.model)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    # retriever를 외부 파일로 저장한 후, (처음 한 번만 만들어 놓기)
    # vectorstore.save_local("JingHaiRetriever")
    
    # 나중에 이것을 불러들여 수행할 수 있음 (아래 방법)
    vectorstore = FAISS.load_local("JingHaiRetriever", cached_embeddings, allow_dangerous_deserialization=True)
    print("JingHaiRetriever 로드 완료")
    print(vectorstore)
    retriever = vectorstore.as_retriever()
    print('retriever')
    print(retriever)
    
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    if model_name.startswith('gpt-'):
        llm = ChatOpenAI(model_name=model_name, temperature=0.1)
    elif model_name.startswith('claude-'):
        llm = ChatAnthropic(model=model_name, temperature=0.1)
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 소설 파일 로드가 선택되면...
if mode == "MODE1: 海派京派 소설":
    saved_retriever = load_embeddings() # 내장되어있는 TXT 문서 로드(Load Documents)
    saved_chain = create_chain(saved_retriever, model_name=selected_model)
    st.session_state["saved_chain"] = saved_chain
    print('사전 저장된 retriever 기반 chain 생성 완료')
    print(saved_chain)

# 파일이 업로드 되었을 때
if mode == "MODE2: 업로드한 파일":
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    uploaded_retriever = embed_file(uploaded_file)
    if uploaded_retriever is not None:
        uploaded_chain = create_chain(uploaded_retriever, model_name=selected_model)
        st.session_state["uploaded_chain"] = uploaded_chain
        print('업로드 파일 기반 chain 생성 완료')
        print(uploaded_chain)
    else: warning_msg.error("파일을 업로드 해주세요.")

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    if mode == "MODE2: 업로드한 파일":
        chain = st.session_state.get("uploaded_chain")
    else:
        chain = st.session_state.get("saved_chain")
    # chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ''
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
        # add_message("assistant", split_documents[0].text) # 어떻게 해야 localhost에서 이것을 print할 수 있을까?
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
