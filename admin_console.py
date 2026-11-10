import os
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="AI Customer Service Agent Admin Console",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for clean enterprise dashboard styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {"id": "TCK-1001", "customer": "Alice Smith", "issue": "Order delayed in transit", "status": "Escalated", "priority": "High", "channel": "E-commerce"},
        {"id": "TCK-1002", "customer": "Bob Jones", "issue": "Billing discrepancy on statement", "status": "Resolved", "priority": "Medium", "channel": "Banking"},
        {"id": "TCK-1003", "customer": "Charlie Davis", "issue": "SIM card activation failure", "status": "Open", "priority": "Urgent", "channel": "Telecom"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your AI Customer Service Agent. How can I help you with your E-commerce order, Banking account, or Telecom service today?"}
    ]

# Sidebar Navigation
with st.sidebar:
    st.title("🤖 Motiff AI Console")
    st.caption("Customer Support AI Agent Admin")
    st.divider()
    
    selected_tab = st.radio(
        "Navigation",
        ["Dashboard Overview", "Customer Support Chatbot", "Policy & Knowledge Base", "Escalations & Tickets", "Agent Analytics"]
    )
    
    st.divider()
    st.markdown("**System Status:** 🟢 Operational")
    st.markdown("**Active Models:** GPT-4o Mini / RAG Engine")

# 1. Dashboard Overview
if selected_tab == "Dashboard Overview":
    st.header("📊 Admin Dashboard Overview")
    st.markdown("Real-time telemetry and performance metrics for your multi-industry AI Customer Service Agent.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Queries Today", value="1,428", delta="+12%")
    with col2:
        st.metric(label="Auto-Resolution Rate", value="84.5%", delta="+3.2%")
    with col3:
        st.metric(label="Escalation Rate", value="15.5%", delta="-1.1%")
    with col4:
        st.metric(label="Avg Resolution Time", value="42s", delta="-8s")
        
    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Query Distribution by Industry")
        chart_data = {"E-commerce": 55, "Banking": 30, "Telecom": 15}
        st.bar_chart(chart_data)
        
    with col_b:
        st.subheader("⚡ Recent Escalations")
        for t in st.session_state.tickets:
            if t["status"] in ["Escalated", "Open"]:
                st.warning(f"**{t['id']}** ({t['channel']}) - {t['customer']}: {t['issue']} [Priority: {t['priority']}]")

# 2. Customer Support AI Chatbot Simulator
elif selected_tab == "Customer Support AI Chatbot":
    st.header("💬 Customer Support AI Chatbot Simulator")
    st.markdown("Test the AI agent's capability to handle FAQs, retrieve customer policies, check order status, and trigger escalations.")
    
    col_sim1, col_sim2 = st.columns([2, 1])
    with col_sim1:
        industry_context = st.selectbox("Select Industry Domain Context", ["E-commerce (Order Tracking & Returns)", "Banking (Account & Transactions)", "Telecom (Plan & Billing)"])
    with col_sim2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = [{"role": "assistant", "content": f"Hello! AI agent initialized for {industry_context}. How can I assist you?"}]
            st.rerun()

    st.divider()

    # Render Chat Messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input & Automated Agent Logic
    if user_input := st.chat_input("Ask about orders, bank statements, or policies (e.g., 'Where is order #ORD-9921?' or 'Escalate to human agent')..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI Agent processing request & checking backend systems..."):
                query_lower = user_input.lower()
                
                if "order" in query_lower or "shipping" in query_lower or "status" in query_lower:
                    response_text = "📦 **Order Status Check:** Found order `#ORD-9921`. Status: **Out for Delivery** with Express Courier. Estimated arrival: Today by 4:00 PM."
                elif "policy" in query_lower or "return" in query_lower or "refund" in query_lower:
                    response_text = "📄 **Policy Retrieval:** According to our E-commerce Return Policy, items can be returned within 30 days of delivery in unused condition with original packaging. Refunds are processed within 3-5 business days."
                elif "escalate" in query_lower or "human" in query_lower or "agent" in query_lower or "speak to" in query_lower:
                    response_text = "⚠️ **Issue Escalated:** I have created escalation ticket `#TCK-1004` and transferred your conversation context to a senior human support specialist. A representative will join this chat within 2 minutes."
                    st.session_state.tickets.append({"id": "TCK-1004", "customer": "Current User", "issue": user_input, "status": "Escalated", "priority": "High", "channel": "E-commerce"})
                else:
                    response_text = f"🤖 **AI Agent Answer:** Based on our knowledge base for **{industry_context}**, I understand you are inquiring about: *'{user_input}'*. Would you like me to check your account details or escalate this to a human agent?"

                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# 3. Policy & Knowledge Base Management
elif selected_tab == "Policy & Knowledge Base":
    st.header("📚 Policy & Knowledge Base Management")
    st.markdown("Upload and manage vector embeddings, documents, and FAQs for RAG retrieval.")
    
    uploaded_policy = st.file_uploader("Upload Company Policy or FAQ Document (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    if uploaded_policy:
        st.success(f"Successfully uploaded and indexed `{uploaded_policy.name}` into vector store!")
        
    st.subheader("Active Knowledge Base Documents")
    kb_data = [
        {"Title": "E-commerce Return & Refund Policy 2026", "Type": "PDF", "Chunks": 42, "Last Updated": "2026-01-15"},
        {"Title": "Banking Fraud & Dispute Resolution Guide", "Type": "PDF", "Chunks": 88, "Last Updated": "2026-02-10"},
        {"Title": "Telecom Roaming & Plan FAQs", "Type": "Markdown", "Chunks": 19, "Last Updated": "2026-03-01"}
    ]
    st.table(kb_data)

# 4. Escalations & Tickets
elif selected_tab == "Escalations & Tickets":
    st.header("🚨 Human Escalation & Ticket Queue")
    st.markdown("Manage customer conversations escalated by the AI agent.")
    
    for t in st.session_state.tickets:
        with st.expander(f"Ticket {t['id']} - {t['customer']} ({t['channel']}) - Status: {t['status']}"):
            st.write(f"**Issue Description:** {t['issue']}")
            st.write(f"**Priority:** {t['priority']}")
            new_status = st.selectbox(
                f"Update Status for {t['id']}", 
                ["Open", "In Progress", "Resolved", "Escalated"], 
                index=["Open", "In Progress", "Resolved", "Escalated"].index(t['status'])
            )
            if st.button(f"Save Status {t['id']}"):
                t['status'] = new_status
                st.success(f"Ticket {t['id']} updated to {new_status}!")

# 5. Agent Analytics
elif selected_tab == "Agent Analytics":
    st.header("📈 AI Agent Analytics & Performance")
    st.markdown("Analyze customer sentiment, intent recognition accuracy, and deflection rates.")
    
    col_an1, col_an2 = st.columns(2)
    with col_an1:
        st.subheader("Customer Sentiment Analysis")
        st.progress(0.85, text="Positive Sentiment (85%)")
        st.progress(0.10, text="Neutral Sentiment (10%)")
        st.progress(0.05, text="Negative Sentiment - Triggered Escalation (5%)")
        
    with col_an2:
        st.subheader("Top Resolved Intent Categories")
        intent_data = {"Order Status": 45, "Billing & Payments": 25, "Plan Changes": 18, "Returns": 12}
        st.bar_chart(intent_data)