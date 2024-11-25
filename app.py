import streamlit as st 
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

if 'topic_model_response' not in st.session_state:
    st.session_state.topic_model_response = ''
if 'in_depth_model_response' not in st.session_state:
    st.session_state.in_depth_model_response = ''

model_choice = st.sidebar.radio('Select a model', ['Topic Model', 'In-Depth Model'])


def get_model_answer(model_name, topic):
    if model_name == 'Topic Model':
        prompt = Topic_Model_prompt(topic)
        model = ChatGroq(model='gemma2-9b-it')
        return model.invoke(prompt)
    
    elif model_name == 'In-Depth Model':
        prompt = In_Depth_Model_prompt(topic)
        model = ChatGroq(model='llama3-groq-70b-8192-tool-use-preview')
        return model.invoke(prompt)


def Topic_Model_prompt(topic):
    template = """
        You are an assistant that helps organize learning paths by identifying key topics and their subtopics for the topic "{topic}". 
        The user wants to learn about {topic}. 
        Please provide a structured list of the main topics and relevant subtopics that the user should explore to gain a comprehensive understanding of {topic}. 

        Organize the topics in a hierarchical format:

        1. Topic 1 /n
        - Subtopic 1.1 /n
        - Subtopic 1.2 /n/n
        2. Topic 2 /n
        - Subtopic 2.1 /n
        - Subtopic 2.2 /n

        Keep the structure clear, concise, and easy to follow.
    """
    prompt_template = PromptTemplate(
        input_variables=['topic'],
        template=template
    )
    return prompt_template.format(topic=topic)


def In_Depth_Model_prompt(topic):
    template = """
        Please provide a detailed explanation of the topic '{topic}'.
        Structure the sections with clear headings and add two new lines after each main section for readability.

        Structure as follows:
        Main Topic:(bold)
        
        Subtopic 1: \n
            Section Title:
            Content goes here...
            
            'Code example here (if necessary)'
           'Reference Link: https://example.com'
        
        Subtopic 2: \n
            Section Title:
            Content goes here...
            
            'Code example here (if necessary)'
            'Reference Link: https://example.com'
        
        give some spacing and do font size according to tipic and subtopics

        Each subtopic should have a broad answer with a clear explanation of its significance, core concepts, practical applications, challenges, and future trends, with code samples where relevant.
    """
    prompt_template = PromptTemplate(
        input_variables=['topic'],
        template=template
    )
    return prompt_template.format(topic=topic)


if model_choice == 'Topic Model':
    st.title("🎓 Welcome To Personal Tutor! 📚")
    topic = st.text_input("Enter the Topic You Want to Learn:")
    if st.sidebar.button("Get the Topics"):
        response = get_model_answer('Topic Model', topic)
        st.session_state.topic_model_response = response.content 
    st.write(st.session_state.topic_model_response)

elif model_choice == 'In-Depth Model':
    st.title("🎓 Welcome To Personal Tutor! 📚")
    sub_topic = st.text_area("Copy Paste the Sub-topic to go In-Depth")
    if st.sidebar.button("Go In-Depth!"):
        response = get_model_answer('In-Depth Model', sub_topic)
        st.session_state.in_depth_model_response = response.content 
    st.write(st.session_state.in_depth_model_response)

if st.sidebar.button('Clear'):
    st.session_state.topic_model_response = ''
    st.session_state.in_depth_model_response = ''
