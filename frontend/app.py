import streamlit as st
import requests

API_URL = "https://islamic-rag-o50w.onrender.com/ask"
st.set_page_config(page_title="Islamic RAG MVP", layout="centered")
st.title("Islamic Knowledge RAG Prototype")
st.caption("Grounded QA using Jamia Binoria Fatawa Subset")

query = st.text_area("Enter your question (Urdu / English):", placeholder="مثال: جنت الفردوس میں اعلی مقام ہے یا نہیں؟")

if st.button("Search & Verify", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying backend...."):
            try:
                response = requests.post(API_URL, json={"query": query}, timeout=90)
                
                # Check status code first before parsing JSON
                if response.status_code != 200:
                    st.error(f"Server Error {response.status_code}: {response.text}")
                else:
                    res = response.json()
                    st.subheader("Answer")
                    st.write(res.get("answer"))

                    st.subheader("Verified Sources")
                    sources = res.get("sources", [])
                    if not sources:
                        st.info("No relevant sources matched.")
                    for src in sources:
                        with st.expander(f"Fatwa #{src.get('fatwa_number')} — {src.get('category')} ({src.get('sub_category')})"):
                            st.write(src.get("snippet"))
                            if src.get("url"):
                                st.markdown(f"[View Original Fatwa on Binoria]({src.get('url')})")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Render's free tier was likely waking up. Please click 'Search & Verify' once more.")
            except Exception as e:
                st.error(f"Error: {e}")