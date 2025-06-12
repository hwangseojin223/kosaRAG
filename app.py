import os
import streamlit as st 

from dotenv import load_dotenv
from llm import get_ai_message

st.set_page_config(
    page_title='소득세 챗봇',
    page_icon=':guardsman:',
    # layout='wide',
    # initial_sidebar_state='expanded'
)

st.title('Streamlit 기본 예제')
st.caption('소득세에 관련된 모든 것을 답변해 드립니다.')

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])
        
        
if user_question := st.chat_input(placeholder='소득세에 관련 궁금한 내용을 말씀해 주세요.'):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user', 'content':user_question})
    
    with st.spinner('답변을 생성하는 중입니다.'):
        ai_response = get_ai_message(user_question)
        
        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_response)
        st.session_state.message_list.append({'role':'ai', 'content':ai_message})
