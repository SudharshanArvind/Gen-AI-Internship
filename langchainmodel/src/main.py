import streamlit as st
from Langchain_helper import get_qa_chain, create_vector_db

st.title("CUSTOMER SUPPORT CHATBOT 🤖")

st.write("✅ Streamlit app started!")  # Debugging message

if st.button("Create Knowledgebase"):
    st.write("⏳ Creating knowledge base...")
    try:
        create_vector_db()
        st.success("✅ Knowledge base created successfully!")
    except Exception as e:
        st.error(f"❌ Error creating knowledge base: {e}")

question = st.text_input("Ask your question:")

if question:
    st.write("⏳ Processing question...")  # Debugging message
    try:
        chain = get_qa_chain()
        if chain:
            response = chain({"query": question})
            st.header("Answer")
            st.write(response["result"])
        else:
            st.error("❌ Error: QA Chain is not initialized.")
    except Exception as e:
        st.error(f"❌ Error processing question: {e}")