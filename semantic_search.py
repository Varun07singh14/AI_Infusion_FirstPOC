import os
import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()                 # loads everything from .env

client = OpenAI()

# Page configuration
st.set_page_config(
    page_title="Enterprise Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Enterprise Semantic Search Engine")
st.caption("AI-powered vector search across Enterprise IT, Healthcare administration, and Legal compliance documents.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key_input = st.text_input("OpenAI API Key", value=env_api_key, type="password", help="Enter your OpenAI API Key")
    
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        
    st.divider()
    st.header("🏢 Domain Filters")
    selected_domain = st.selectbox(
        "Filter by Industry Sector",
        ["All Domains", "Enterprise IT", "Healthcare Administration", "Legal & Compliance"]
    )
    
    st.divider()
    st.markdown("**Search Settings**")
    top_k = st.slider("Results to retrieve (k)", min_value=1, max_value=10, value=3)

# Initialize Sample Knowledge Base Documents (Cross-Industry)
@st.cache_resource
def initialize_vector_store():
    sample_docs = [
        # Enterprise IT
        Document(
            page_content="Server accounts are automatically locked after 5 consecutive failed authentication attempts. To unlock an account, administrators must verify user identity via multi-factor authentication (MFA) and issue a temporary token valid for 30 minutes through the IT Service Desk portal.",
            metadata={"domain": "Enterprise IT", "source": "IT_Security_Policy_v4.pdf", "id": "IT-101"}
        ),
        Document(
            page_content="Database horizontal sharding configuration requires partitioning multi-tenant data based on tenant_id hash ranges to mitigate hot shard bottlenecks. Redis caching layers must be deployed in cluster mode with a 15-minute Time-To-Live (TTL) for high-frequency lookup endpoints.",
            metadata={"domain": "Enterprise IT", "source": "Database_Architecture_Standard.docx", "id": "IT-102"}
        ),
        
        # Healthcare Administration
        Document(
            page_content="Under HIPAA Security Rule mandates, all electronic Protected Health Information (ePHI) at rest must be encrypted using AES-256 bit encryption standards. Access logs must be retained for a minimum of six years and audited on a quarterly basis by compliance officers.",
            metadata={"domain": "Healthcare Administration", "source": "HIPAA_Compliance_Manual_2026.pdf", "id": "HC-201"}
        ),
        Document(
            page_content="Patient admission triage protocols require recording vital signs, insurance verification, and primary diagnosis code (ICD-10) within 15 minutes of arrival in the emergency department. Uninsured patients must be provided with financial assistance disclosures prior to non-emergency discharge.",
            metadata={"domain": "Healthcare Administration", "source": "Patient_Workflow_Standard.pdf", "id": "HC-202"}
        ),
        
        # Legal & Compliance
        Document(
            page_content="Corporate Insolvency Resolution Process (CIRP) filings under Section 7 require submission of default proofs, certified bank statements, and operational creditor ledger copies within 14 days of NCDRC notice issuance.",
            metadata={"domain": "Legal & Compliance", "source": "Corporate_Litigation_Guide.docx", "id": "LG-301"}
        ),
        Document(
            page_content="Non-Disclosure Agreements (NDAs) executed with third-party vendors must include strict jurisdiction clauses specifying Bangalore courts, perpetual confidentiality obligations regarding source code, and a liquidated damages clause for intellectual property breaches.",
            metadata={"domain": "Legal & Compliance", "source": "Vendor_Contract_Framework.pdf", "id": "LG-302"}
        )
    ]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(sample_docs, embedding=embeddings)
    return vector_store

# Load Vector Store
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to run semantic searches.")
else:
    vector_store = initialize_vector_store()
    
    # Natural Language Search Input
    query = st.text_input("Enter natural-language search query (e.g., 'How do I unlock a server account?' or 'What are the HIPAA encryption requirements?'):")
    
    if query:
        with st.spinner("Executing semantic vector search..."):
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=top_k)
            
            st.markdown(f"### Search Results for: *'{query}'*")
            
            match_count = 0
            for doc, score in results:
                # Apply domain filter if selected
                if selected_domain != "All Domains" and doc.metadata["domain"] != selected_domain:
                    continue
                
                match_count += 1
                # Convert distance score to a visual relevance percentage (approximate)
                relevance = max(0, min(100, int((1 - score / 2.0) * 100)))
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Source:** `{doc.metadata['source']}` | **Domain:** `{doc.metadata['domain']}` | **ID:** `{doc.metadata['id']}`")
                        st.info(doc.page_content)
                    with col2:
                        st.metric(label="Relevance Match", value=f"{relevance}%")
                    st.divider()
            
            if match_count == 0:
                st.warning("No matching documents found within the selected domain filter.")