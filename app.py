import streamlit as st
from retrival import get_response

# Page Configuration
st.set_page_config(
    page_title="Babaji's Portfolio Assistant",
    page_icon="🤖",
    layout="centered"
)

# Optional minimal styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
st.title("🤖 Babaji's Portfolio Assistant")
st.markdown("Ask anything about **Babaji (a.k.a. Arjun)** — his education, projects, experience, and more.")

# Input Box
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("🔍 Ask me a question:")
    submitted = st.form_submit_button("Ask")

# Response
if submitted and question:
    with st.spinner("Thinking..."):
        response = get_response(question)

    st.markdown("---")
    st.markdown("**💡 Answer:**")
    lines = response.split("\n")
    for line in lines:
        if line.strip():
            st.markdown(f"- {line.strip()}")