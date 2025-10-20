# import os
# import streamlit as st
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from io import BytesIO
# load_dotenv()
# api_key = os.getenv('GOOGLE_API_KEY')
# st.set_page_config(
#     page_title="AI College Professor",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# st.title("AI Study Material Generator")
#
# chat_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             (
#                 "You are a world-class college professor. "
#                 "Your primary goal is to make all topics so easy for students that once they learn from your material, they don't need to study again for any exam or interview. "
#                 "You base your study material on the rigor and clarity of great personalities like Andrew Ng. "
#                 "You MUST adhere to this structure: "
#                 "1. **Short Overview:** Start with a brief, high-level overview of the subtopic so a first-timer becomes immediately familiar with the context. "
#                 "2. **Deep Dive with Examples:** Then, dive deep and explain the topic with detailed, Theoretical and  practical examples such that the student never forgets the concept."
#                 "If you are asked to give material in detailed form then make sure that you are including figures as per need and also content is at least 15 A4 size papers long."
#                 "Ensure the output is well-formatted using Markdown."
#             )
#         ),
#         (
#             "user",
#             (
#                 "I am a student in my {year} year, studying the {department} department. "
#                 "I am preparing for exams in the subject: '{subject}'. "
#                 "I want study material for the chapter '{chapter}', specifically covering these topics: {topics}. "
#                 "Please present the material in a {form} format (e.g., lecture notes, detailed summary, Q&A). "
#                 "Generate the material to the highest standard so I can ace my exams."
#             )
#         )
#     ]
# )
#
# try:
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=api_key)
# except Exception as e:
#     st.error(
#         f"Failed to initialize OllamaLLM. Ensure Ollama is running and the 'llama3.2:latest' model is pulled. Error: {e}")
#     st.stop()
#
# with st.sidebar:
#     st.header("Your Study Requirements")
#
#     year = st.selectbox("Current Academic Year:", ["1st", "2nd", "3rd", "4th"], index=2)
#     department = st.text_input("Department:", "Computer Science")
#     subject = st.text_input("Subject Name:", "Introduction to Machine Learning")
#     chapter = st.text_input("Chapter Name:", "Linear Regression")
#     topics = st.text_area("Specific Topics to Cover (Comma-separated):",
#                           "Cost Function, Gradient Descent Algorithm, Feature Scaling")
#     form = st.selectbox("Preferred Format for Study Material:",
#                         ["Detailed Lecture Notes", "Comprehensive Summary", "Q&A Format"], index=0)
#
#     submit_button = st.button("Generate Study Material", type="primary")
#
# st.header("Generated Study Material")
#
#
#
# if submit_button:
#     chat_variables = {
#         "year": year,
#         "department": department,
#         "subject": subject,
#         "chapter": chapter,
#         "topics": topics,
#         "form": form
#     }
#
#     chain = chat_template | llm
#
#     with st.spinner(f"Generating comprehensive material on {chapter} for you..."):
#         try:
#             results = chain.invoke(chat_variables)
#
#             output_text = str(getattr(results, "content", results))
#
#             st.success("Material Ready!")
#
#             st.markdown(output_text, unsafe_allow_html=False)
#
#         except Exception as e:
#             st.error(
#                 f"An error occurred during LLM invocation. Check your Ollama server status or the model name. Details: {e}")


import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from io import BytesIO
import re  # We need the regex library

# --- PDF Conversion Libraries ---
from markdown_it import MarkdownIt
# REMOVED: mdit_py_plugins.texmath and amsmath_plugin
from weasyprint import HTML, CSS

# --------------------------------

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

st.set_page_config(
    page_title="AI College Professor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Study Material Generator")


# =================================================================
# === PDF Conversion Functions with Aggressive LaTeX Cleaning ===
# =================================================================

def replace_simple_latex_with_unicode(text: str) -> str:
    """
    Aggressively cleans the text by replacing LaTeX commands and removing
    LaTeX syntax elements like subscripts, superscripts, and equation delimiters.
    This ensures clean text for WeasyPrint.
    """
    replacements = {
        '\\theta': 'θ', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ',
        '\\delta': 'δ', '\\epsilon': 'ε', '\\pi': 'π', '\\lambda': 'λ',
        '\\sum': '∑', '\\int': '∫', '\\sqrt': '√', '\\pm': '±',
        '\\approx': '≈', '\\cdot': '·',
        # Handle the common case of h_\theta(x) -> h_θ(x)
        # We need a regex for complex structures, but a simple replace helps:
        'h_\\theta(x)': 'h_θ(x)',
        'J(\\theta_0, \\theta_1)': 'J(θ₀, θ₁)',
        '\\theta_0': 'θ₀',
        '\\theta_1': 'θ₁',
        '\\theta_i': 'θᵢ',

        # General cleaning: replace subscripts/superscripts to plain text
        '^{(i)}': '(i)',  # x^{(i)} -> x(i)
    }

    # 1. Apply simple string replacements for symbols
    for latex, unicode_char in replacements.items():
        text = text.replace(latex, unicode_char)

    # 2. Aggressively remove all remaining LaTeX delimiters
    # Remove $ and $$ delimiters
    text = text.replace('$$', '').replace('$', '')

    # 3. Remove any remaining structural LaTeX commands (e.g., \frac, \mathbf, etc.)
    # This is a basic filter; a full parser is needed for perfect conversion.
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # 4. Remove braces that were used for grouping after the LaTeX command is gone
    text = text.replace('{', '').replace('}', '')

    return text


@st.cache_resource
def get_pdf_html_converter():
    """Initializes and returns the markdown-it-py parser without math plugins."""
    # We remove math plugins because the text is pre-cleaned to prevent rendering issues.
    md = (
        MarkdownIt('commonmark', {'breaks': True, 'html': True})
        .enable(['table', 'strikethrough'])  # Enable standard GFM extensions
    )
    return md


def convert_markdown_to_pdf_bytes(markdown_text: str) -> BytesIO:
    """Converts the markdown string (with cleaned LaTeX) into PDF bytes using WeasyPrint."""

    # APPLY THE FIX: Convert simple LaTeX to Unicode before rendering
    cleaned_text = replace_simple_latex_with_unicode(markdown_text)

    md_parser = get_pdf_html_converter()
    html_content = md_parser.render(cleaned_text)

    # 1. HTML Wrapper: Includes minimal CSS for readability and a title.
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Generated Study Material</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #1E90FF; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; border-radius: 4px; overflow-x: auto; }}
            p, ul, ol, li {{ margin-bottom: 1em; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # 2. Convert HTML to PDF Bytes
    pdf_file = BytesIO()
    HTML(string=final_html).write_pdf(pdf_file)
    pdf_file.seek(0)
    return pdf_file


# =================================================================
# === LangChain Setup and Streamlit UI ===
# =================================================================

chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a world-class college professor. "
                "Your primary goal is to make all topics so easy for students that once they learn from your material, they don't need to study again for any exam or interview. "
                "You base your study material on the rigor and clarity of great personalities like Andrew Ng. "
                "You MUST adhere to this structure: "
                "1. **Short Overview:** Start with a brief, high-level overview of the subtopic so a first-timer becomes immediately familiar with the context. "
                "2. **Deep Dive with Examples:** Then, dive deep and explain the topic with detailed, Theoretical and practical examples such that the student never forgets the concept."
                "If you are asked to give material in detailed form then make sure that you are including figures as per need and also content is at least 15 A4 size papers long."
                "Ensure the output is well-formatted using Markdown. When writing math, use the **actual Unicode symbol** for simple inline variables (e.g., θ instead of \\theta) to ensure maximum compatibility across systems."
            )
        ),
        (
            "user",
            (
                "I am a student in my {year} year, studying the {department} department. "
                "I am preparing for exams in the subject: '{subject}'. "
                "I want study material for the chapter '{chapter}', specifically covering these topics: {topics}. "
                "Please present the material in a {form} format (e.g., lecture notes, detailed summary, Q&A). "
                "Generate the material to the highest standard so I can ace my exams."
            )
        )
    ]
)

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=api_key)
except Exception as e:
    st.error(
        f"Failed to initialize ChatGoogleGenerativeAI. Ensure GOOGLE_API_KEY is set correctly. Error: {e}")
    st.stop()

with st.sidebar:
    st.header("Your Study Requirements")

    year = st.selectbox("Current Academic Year:", ["1st", "2nd", "3rd", "4th"], index=2)
    department = st.text_input("Department:", "Computer Science")
    subject = st.text_input("Subject Name:", "Introduction to Machine Learning")
    chapter = st.text_input("Chapter Name:", "Linear Regression")
    topics = st.text_area("Specific Topics to Cover (Comma-separated):",
                          "Cost Function, Gradient Descent Algorithm, Feature Scaling")
    form = st.selectbox("Preferred Format for Study Material:",
                        ["Detailed Lecture Notes", "Comprehensive Summary", "Q&A Format"], index=0)

    submit_button = st.button("Generate Study Material", type="primary")

st.header("Generated Study Material")

if submit_button:
    chat_variables = {
        "year": year,
        "department": department,
        "subject": subject,
        "chapter": chapter,
        "topics": topics,
        "form": form
    }

    chain = chat_template | llm

    with st.spinner(f"Generating comprehensive material on {chapter} for you... 🧠"):
        try:
            results = chain.invoke(chat_variables)
            output_text = str(getattr(results, "content", results))

            st.success("Material Ready! Scroll down to download the PDF.")

            # 1. Display the material using Streamlit's markdown renderer
            st.markdown(output_text, unsafe_allow_html=False)

            st.markdown("---")
            st.subheader("Download as PDF 💾")

            # 2. Convert the output to PDF bytes (Unicode conversion happens inside here)
            pdf_bytes = convert_markdown_to_pdf_bytes(output_text)

            # 3. Create the download button
            st.download_button(
                label="Download Study Material as PDF",
                data=pdf_bytes,
                file_name=f"{chapter}_Study_Material.pdf",
                mime="application/pdf",
                type="secondary"
            )

        except Exception as e:
            st.error(
                f"An error occurred during material generation. Details: {e}")