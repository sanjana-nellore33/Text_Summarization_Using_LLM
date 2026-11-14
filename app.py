
import os
import streamlit as st
from dotenv import load_dotenv

from utils.preprocessing import (
    extract_pdf_text,
    clean_text,
    split_into_chunks
)

from utils.summarizer import summarize_text


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📄",
    layout="wide"
)

# Application heading
st.title("📄 AI Text Summarization Application")

st.write(
    "Upload a PDF document or paste text to generate "
    "an intelligent summary using Google Gemini AI."
)

st.divider()

# Choose input method
input_method = st.radio(
    "Choose document input method:",
    ["Upload PDF", "Enter Text"],
    horizontal=True
)

document_text = ""

# PDF upload section
if input_method == "Upload PDF":

    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:
            document_text = extract_pdf_text(uploaded_file)

            if document_text.strip():
                st.success("PDF uploaded and text extracted successfully!")
            else:
                st.warning(
                    "No readable text was found in this PDF. "
                    "It may contain scanned images."
                )

        except Exception as error:
            st.error(f"Could not read the PDF: {error}")

# Manual text section
else:

    document_text = st.text_area(
        "Paste your document text here:",
        height=250,
        placeholder="Enter your document content..."
    )

# Summary settings
st.divider()

st.subheader("Summary Settings")

summary_length = st.selectbox(
    "Choose summary length:",
    ["Short", "Medium", "Detailed"]
)

# Generate summary
if st.button("Generate Summary", type="primary"):

    if not os.getenv("GEMINI_API_KEY"):

        st.error(
            "Gemini API key is missing. Please check your .env file."
        )

    elif not document_text.strip():

        st.warning("Please upload a PDF or enter some text first.")

    else:

        try:

            with st.spinner(
                "Processing your document and generating summary..."
            ):

                processed_text = clean_text(document_text)

                chunks = split_into_chunks(processed_text)

                summary = summarize_text(
                    chunks,
                    summary_length
                )

            st.success("Summary generated successfully!")

            # Document statistics
            st.subheader("Document Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Original Words",
                len(processed_text.split())
            )

            col2.metric(
                "Text Chunks",
                len(chunks)
            )

            col3.metric(
                "Summary Words",
                len(summary.split())
            )

            # Summary output
            st.divider()

            st.subheader("Generated Summary")

            st.write(summary)

            # Download option
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="generated_summary.txt",
                mime="text/plain"
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )

st.divider()

st.caption(
    "Built using Python, Streamlit and Google Gemini API."
)
