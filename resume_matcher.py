import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()                 # loads everything from .env

client = OpenAI()

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening & Job Matching",
    page_icon="📋",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("📋 AI Resume Screening & Job Matching Engine")
st.caption("Recruitment, Staffing & HR Tech — Automated skill extraction, JD parsing, and evidence-based candidate reporting.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key_input = st.text_input("OpenAI API Key", value=env_api_key, type="password", help="Enter your OpenAI API Key")
    
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        
    st.divider()
    st.header("⚡ Quick Load Samples")
    load_samples = st.button("Load Sample JD & Resume")

# Default Sample Data
sample_jd = """Job Title: Senior AI & Backend Engineer
Location: Bengaluru / Remote (Wells Fargo Tech Hub)
Requirements:
- 5+ years of experience in Python software development and distributed systems.
- Strong expertise in Large Language Models (LLMs), LangChain, vector databases (Chroma, FAISS), and RAG pipelines.
- Deep understanding of database engineering (sharding, caching layers like Redis/Caffeine).
- Experience building REST APIs and microservice architectures.
- Bachelor's or Master's degree in Computer Science or related technical field."""

sample_resume = """Varun Kumar Singh
Engineering Lead & Vice President | Bengaluru, India
Summary:
Engineering leader with 12+ years of experience specializing in distributed systems, backend architecture, and AI engineering.
Technical Skills:
- Programming Languages: Python, JavaScript
- AI & GenAI: LangChain, OpenAI SDK, Groq API, Retrieval-Augmented Generation (RAG), Vector Databases, Text Chunking Strategies
- Distributed Systems & Databases: Apache Kafka (KRaft, brokers, DLQ), horizontal/vertical sharding, Redis, Caffeine caching layers
- Frameworks & Tools: FastAPI, Flask, Docker, Git, Windows/Linux environments
Experience:
- Engineering Lead & Vice President at Wells Fargo (2026): Leading backend engineering teams, designing high-throughput transaction pipelines and database query optimization.
- Scaler AI Infusion Curriculum: Advanced studies in AI engineering development scripts, transformer architecture mechanics, self-attention, and vector search implementation."""

# Initialize session state for inputs if load samples clicked
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if load_samples:
    st.session_state.jd_text = sample_jd
    st.session_state.resume_text = sample_resume

# Main Layout - Two Columns for JD and Resume Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Description (JD)")
    jd_input = st.text_area(
        "Paste Job Description requirements here:", 
        value=st.session_state.jd_text, 
        height=300,
        key="jd_textarea"
    )

with col2:
    st.subheader("📄 Candidate Resume")
    resume_input = st.text_area(
        "Paste Candidate Resume text here:", 
        value=st.session_state.resume_text, 
        height=300,
        key="resume_textarea"
    )

st.divider()

# Matching Execution Button
if st.button("🚀 Run AI Resume Match & Generate Report", type="primary"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar.")
    elif not jd_input.strip() or not resume_input.strip():
        st.error("⚠️ Please provide both a Job Description and a Candidate Resume.")
    else:
        with st.spinner("Analyzing resume, mapping skills, and generating evidence-based match report..."):
            
            # Setup LLM & Prompt
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an expert HR Technology and Technical Recruiting AI Agent. "
                    "Analyze the provided Job Description and Candidate Resume. "
                    "Generate a rigorous, evidence-based match report structured exactly as follows:\n\n"
                    "1. **Extracted Candidate Skills & Experience Summary**\n"
                    "2. **JD Key Requirement Alignment** (Matching vs Missing requirements)\n"
                    "3. **Overall Match Score** (Percentage from 0% to 100% with justification)\n"
                    "4. **Detailed Gap Analysis**\n"
                    "5. **Hiring Recommendation** (Strong Hire / Interview / Reject with actionable reasoning)"
                )),
                ("human", "Job Description:\n{jd}\n\nCandidate Resume:\n{resume}")
            ])
            
            chain = prompt_template | llm
            response = chain.invoke({"jd": jd_input, "resume": resume_input})
            
            # Display Results
            st.success("Analysis complete!")
            st.markdown("### 📊 Evidence-Based Match Report")
            st.markdown(response.content)