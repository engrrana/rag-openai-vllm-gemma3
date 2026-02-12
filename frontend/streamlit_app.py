import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="UET Lahore Chatbot",
    page_icon="🎓",
    layout="centered"
)

# API URL
API_URL = "http://localhost:8000"

# Title
st.title("🎓 UET Lahore Department Assistant")
st.markdown("Ask me anything about UET Lahore departments, programs, and eligibility!")

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API Online")
            st.info(f"📚 {data.get('total_documents', 0)} documents loaded")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.warning("Start backend: `python run.py`")
    
    st.divider()
    
    # Session stats
    st.header("Session Stats")
    if "messages" in st.session_state:
        st.metric("Total Messages", len(st.session_state.messages))
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Questions Asked", user_msgs)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Sample questions
    st.header("💡 Sample Questions")
    st.markdown("""
    - What programs does Computer Science offer?
    - Who is the chairman of Electrical Engineering?
    - Eligibility for M.Sc. Data Science?
    - Tell me about Physics department
    - What is offered by Mechanical Engineering?
    """)
    
    st.divider()
    
    # About
    st.header("About")
    st.info("""
    **Powered by:**
    - OpenAI Embeddings
    - ChromaDB
    - FastAPI
    - Streamlit
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm your UET Lahore Department Assistant. I can help you with information about departments, programs, eligibility criteria, and faculty. What would you like to know?"
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    metadata = source.get("metadata", {})
                    content = source.get("content", "")
                    
                    st.markdown(f"**Source {i}**")
                    st.markdown(f"**Department:** {metadata.get('department_name', 'N/A')}")
                    st.markdown(f"**Section:** {metadata.get('section', 'N/A')}")
                    st.markdown(f"**Content:** {content[:150]}...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about departments, programs, eligibility..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Try to get answer from API
                response = requests.post(
                    f"{API_URL}/api/v1/answer",
                    json={"question": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer available.")
                    sources = data.get("source_documents", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                metadata = source.get("metadata", {})
                                content = source.get("content", "")
                                
                                st.markdown(f"**Source {i}**")
                                st.markdown(f"**Department:** {metadata.get('department_name', 'N/A')}")
                                st.markdown(f"**Section:** {metadata.get('section', 'N/A')}")
                                st.markdown(f"**Content:** {content[:150]}...")
                                st.divider()
                    
                    st.rerun()
                
                elif response.status_code == 429:
                    # OpenAI quota exceeded - try retrieve endpoint
                    st.warning("⚠️ OpenAI quota exceeded. Trying document retrieval...")
                    
                    retrieve_response = requests.post(
                        f"{API_URL}/api/v1/retrieve",
                        json={"question": prompt},
                        timeout=10
                    )
                    
                    if retrieve_response.status_code == 200:
                        data = retrieve_response.json()
                        documents = data.get("retrieved_documents", [])
                        
                        if documents:
                            answer = "**Here are the relevant documents:**\n\n"
                            for i, doc in enumerate(documents, 1):
                                metadata = doc.get("metadata", {})
                                content = doc.get("content", "")
                                answer += f"**{i}. {metadata.get('department_name', 'N/A')} - {metadata.get('section', 'N/A')}**\n\n"
                                answer += f"{content}\n\n"
                            
                            st.markdown(answer)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": documents
                            })
                            
                            st.rerun()
                        else:
                            error_msg = "No relevant documents found. Please try rephrasing."
                            st.info(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.rerun()
                    else:
                        error_msg = "Could not retrieve documents. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.rerun()
                
                else:
                    error_msg = f"API Error (Status {response.status_code})"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()
                    
            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to backend API. Make sure the server is running:\n\n`python run.py`"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
                
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()