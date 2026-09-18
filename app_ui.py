import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

st.title("📄 DocuMind AI - پردازش و پرسش از مستندات")
st.caption("سامانه هوشمند خلاصه‌سازی و RAG بر پایه TF-IDF و Cosine Similarity")

# ذخیره متن مستند در Session State
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

with st.sidebar:
    st.header("📥 بارگذاری مستند")
    option = st.radio("روش ورود متن:", ["آپلود فایل PDF", "ورود متن دستی"])
    
    if option == "آپلود فایل PDF":
        uploaded_file = st.file_uploader(
            "فایل PDF خود را انتخاب کنید", 
            type=["pdf"], 
            key="pdf_uploader"
        )
        if uploaded_file and st.button("پردازش PDF", type="primary", key="btn_process_pdf"):
            file_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
            
            with st.spinner("در حال استخراج متن..."):
                try:
                    res = requests.post(f"{API_URL}/extract-text", files=files)
                    if res.status_code == 200:
                        st.session_state.document_text = res.json()["text"]
                        st.success("متن با موفقیت استخراج شد!")
                    else:
                        st.error(f"خطای سرور ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"عدم اتصال به سرور بک‌اند: {e}")
    else:
        manual_text = st.text_area("متن را اینجا وارد کنید:", height=200, key="manual_text_area")
        if st.button("ثبت متن", type="primary", key="btn_submit_manual"):
            if manual_text.strip():
                st.session_state.document_text = manual_text
                st.success("متن با موفقیت ثبت شد!")
            else:
                st.warning("لطفاً متنی وارد کنید.")

# نمایش محتوای اصلی
if st.session_state.document_text:
    tab1, tab2 = st.tabs(["📌 خلاصه‌سازی هوشمند", "💬 پرسش و پاسخ (Q&A)"])
    
    with tab1:
        st.subheader("خلاصه مستند")
        if st.button("تولید خلاصه", key="btn_generate_summary"):
            with st.spinner("در حال تحلیل متن..."):
                try:
                    res = requests.post(
                        f"{API_URL}/summarize", 
                        json={"text": st.session_state.document_text}
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.summary = data["summary"]
                        st.info(f"تعداد کل تکه‌ها (Chunks): {data['total_chunks']}")
                    else:
                        st.error("خطا در خلاصه‌سازی.")
                except Exception as e:
                    st.error(f"خطا در ارتباط با سرور: {e}")
        
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    with tab2:
        st.subheader("سوال خود را درباره متن بپرسید")
        question = st.text_input("سوال:", key="input_question")
        if st.button("جستجو و پاسخ", key="btn_search_answer"):
            if question.strip():
                with st.spinner("در حال پیدا کردن پاسخ مرتبط..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/query",
                            json={
                                "question": question,
                                "document_text": st.session_state.document_text
                            }
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.write("### پاسخ سیستم:")
                            st.write(data["answer"])
                        else:
                            st.error("خطا در پردازش سوال.")
                    except Exception as e:
                        st.error(f"خطا در ارتباط با سرور: {e}")
            else:
                st.warning("لطفاً یک سوال وارد کنید.")
else:
    st.info("👈 لطفا ابتدا از منوی سمت چپ یک فایل PDF آپلود کنید یا متنی وارد نمایید.")