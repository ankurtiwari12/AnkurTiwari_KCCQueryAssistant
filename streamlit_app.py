import streamlit as st
import requests
import pandas as pd
import json

# Configure the page
st.set_page_config(
    page_title="KCC Dataset LLM Interface",
    page_icon="🤖",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model-info")
        return response.json()
    except:
        return None

def generate_response(prompt, system_prompt=None, use_rag=True, k=3):
    try:
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "use_rag": use_rag,
            "k": k
        }
        response = requests.post(f"{API_URL}/generate", json=payload)
        return response.json()
    except Exception as e:
        return {"response": f"Error: {str(e)}", "timing": {}}

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    return f"{seconds:.2f} s"

# Main UI
st.title("KCC Dataset LLM Interface")

# Sidebar
with st.sidebar:
    st.header("Model Information")
    if check_api_health():
        st.success("API Status: Online")
        model_info = get_model_info()
        if model_info:
            st.json(model_info)
    else:
        st.error("API Status: Offline")
        st.warning("Please make sure the API server is running")
    
    # RAG Settings
    st.header("RAG Settings")
    use_rag = st.checkbox("Use RAG", value=True)
    k_value = st.slider("Number of relevant documents", min_value=1, max_value=10, value=3)

# Main content
st.header("Generate Responses")

# Input form
with st.form("generation_form"):
    prompt = st.text_area("Enter your prompt:", height=100)
    system_prompt = st.text_area("System prompt (optional):", height=50)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.form_submit_button("Generate")

# Handle form submission
if submit_button:
    if not prompt:
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating response..."):
            result = generate_response(prompt, system_prompt, use_rag, k_value)
            response = result.get("response", "")
            timing = result.get("timing", {})
            
        st.header("Generated Response")
        st.write(response)
        
        # Display timing information
        st.header("Timing Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Time", format_time(timing.get('total_time', 0)))
        
        with col2:
            if use_rag:
                st.metric("RAG Retrieval Time", format_time(timing.get('rag_retrieval_time', 0)))
        
        with col3:
            st.metric("Generation Time", format_time(timing.get('generation_time', 0)))
        
        # Save conversation history
        if "history" not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "use_rag": use_rag,
            "k_value": k_value,
            "response": response,
            "timing": timing
        })

# Display conversation history
if "history" in st.session_state and st.session_state.history:
    st.header("Conversation History")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Conversation {len(st.session_state.history) - i}"):
            st.write("**Prompt:**")
            st.write(item["prompt"])
            if item["system_prompt"]:
                st.write("**System Prompt:**")
                st.write(item["system_prompt"])
            st.write("**RAG Settings:**")
            st.write(f"- RAG Enabled: {item['use_rag']}")
            st.write(f"- Number of documents: {item['k_value']}")
            st.write("**Response:**")
            st.write(item["response"])
            st.write("**Timing:**")
            timing = item["timing"]
            st.write(f"- Total Time: {format_time(timing.get('total_time', 0))}")
            if item['use_rag']:
                st.write(f"- RAG Retrieval Time: {format_time(timing.get('rag_retrieval_time', 0))}")
            st.write(f"- Generation Time: {format_time(timing.get('generation_time', 0))}") 