import streamlit as st
import requests

API_BASE_URL = "https://islamic-rag-o50w.onrender.com"
st.set_page_config(page_title="Islamic RAG MVP", layout="centered")
st.title("Islamic Knowledge RAG Prototype")
st.caption("Grounded QA using Jamia Binoria Fatawa Subset")

content_type = st.selectbox("Search in:", ["Fatwas", "Sunnah"])
output_language = st.selectbox(
    "Answer language:",
    options=["ur", "en"],
    format_func=lambda language: "اردو" if language == "ur" else "English",
)
query = st.text_area("Enter your question (Urdu / English):", placeholder="مثال: جنت الفردوس میں اعلی مقام ہے یا نہیں؟")

if st.button("Search & Verify", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying backend...."):
            try:
                endpoint = "/ask" if content_type == "Fatwas" else "/ask/sunnah"
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json={"query": query, "output_language": output_language},
                    timeout=90,
                )
                
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
                        source_title = (
                            f"Fatwa #{src.get('fatwa_number')} — {src.get('category')} "
                            f"({src.get('sub_category')})"
                            if src.get("fatwa_number")
                            else src.get("section", "Sunnah reference")
                        )
                        with st.expander(source_title):
                            st.write(src.get("snippet"))
                            if src.get("url"):
                                st.markdown(f"[View Original Fatwa on Binoria]({src.get('url')})")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Render's free tier was likely waking up. Please click 'Search & Verify' once more.")
            except Exception as e:
                st.error(f"Error: {e}")