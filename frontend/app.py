import streamlit as st
import requests

st.set_page_config(page_title="Islamic RAG MVP", layout="centered")
st.title("Islamic Knowledge RAG Prototype")
st.caption("Grounded QA using Jamia Binoria Fatawa Subset")

query = st.text_area("Enter your question (Urdu / English):", placeholder="مثال: جنت الفردوس میں اعلی مقام ہے یا نہیں؟")

if st.button("Search & Verify", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                response = requests.post("http://127.0.0.1:8000/ask", json={"query": query}, timeout=60)
                res = response.json()

                if response.status_code != 200:
                    st.error(f"API Error ({response.status_code}): {res.get('detail', res)}")
                else:
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
            except Exception as e:
                st.error(f"Connection Error: {e}")