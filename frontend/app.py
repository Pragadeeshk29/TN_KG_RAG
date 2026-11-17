import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TN Agriculture KG-RAG", page_icon="🌾")
st.title("🌾 Tamil Nadu Agriculture Schemes KG-RAG")

st.markdown("Ask questions about Tamil Nadu Agriculture - Farmers Welfare schemes.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest source into Neo4j AuraDB"):
        with st.spinner("Fetching source, embedding chunks, and building graph..."):
            r = requests.post(f"{BACKEND_URL}/ingest")
            if r.status_code == 200:
                data = r.json()
                st.success(data["message"])
                st.info(f"Schemes indexed: {data['schemes_indexed']}")
            else:
                st.error(r.text)

with col2:
    if st.button("Preview graph"):
        r = requests.get(f"{BACKEND_URL}/graph-preview")
        if r.status_code == 200:
            st.json(r.json())
        else:
            st.error(r.text)

st.subheader("Suggested questions")
st.caption("- What schemes are listed for the Agriculture - Farmers Welfare Department?")
st.caption("- Which schemes mention seeds?")
st.caption("- Show agriculture schemes related to training for farmers.")
st.caption("- What Tamil Nadu farmer schemes in the source mention soil or gypsum?")

question = st.chat_input("Ask a question about TN agriculture schemes")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question, "top_k": 5}
        )

    if r.status_code == 200:
        data = r.json()
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": data["answer"],
            "sources": data.get("sources", [])
        })
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Error: {r.text}",
            "sources": []
        })

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources**")
            for s in msg["sources"]:
                st.caption(
                    f"{s.get('scheme_name')} | {s.get('source_url')} | "
                    f"{s.get('page')} | score={s.get('score')}"
                )
