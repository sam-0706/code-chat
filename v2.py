import streamlit as st
from ctransformers import AutoModelForCausalLM

# Load the model
llm = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id="mistral-7b-instruct-v0.2.Q2_K.gguf",
    model_type="mistral",
)

st.title("Conversational Chat with Mistral 🦙🗨️")


# Function to generate response
def generate_response(user_query):
    prompt = f"""The user query is '{user_query}'"""
    args = {
        "prompt": prompt,
        "stream": True,
        "max_new_tokens": 2048,
        "temperature": 0,
    }

    response_placeholder = st.empty()  # Placeholder for displaying response chunks

    response_so_far = ""  # Initialize empty string to store cumulative response

    for chunk in llm(**args):
        response_so_far += chunk  # Append current chunk to cumulative response
        response_placeholder.write(response_so_far)  # Display cumulative response

    return  # No need to return anything


# User input
user_query = st.text_input("Enter your query:", "")

if user_query:
    # Generate and display response
    generate_response(user_query)
