import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from educhain import Educhain, LLMConfig

# Streamlit Dark Theme
st.set_page_config(
    page_title="Educhain MCQ Generator from PDF",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
    }
     [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    [data-testid="stToolbar"] {
        right: 2rem;
    }
    [data-testid="stTextInput"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
     [data-testid="stNumberInput"] > div > div > div {
        background-color: #1f2937;
        color: white;
         }
    [data-testid="stNumberInput"] input {
        color: white; /* Fix number input text color */
        }
    [data-testid="stButton"] button {
        background-color: #4CAF50;
        color: white;
    }
    [data-testid="stTextArea"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
    [data-testid="stSelectbox"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
     [data-testid="stSidebar"] {
        background-color: #1f2937;
        color: white;
    }
    [data-testid="stSidebar"] h1, h2, h3, h4, h5, h6, p, ul, ol, li {
        color: white;
    }
    [data-testid="stMarkdownContainer"] h1, h2, h3, h4, h5, h6, p, ul, ol, li {
        color: white;
    }
    [data-testid="stMarkdownContainer"] {
        color: white;
    }
   </style>
    """,
    unsafe_allow_html=True,
)

st.title("Educhain MCQ Generator from PDF")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox("Choose LLM Provider", ["Gemini", "OpenRouter"])
    num_questions_input = st.number_input("Number of Questions", min_value=1, max_value=1000, value=3, step=1)

    # API Key input
    if llm_provider == "Gemini":
        gemini_api_key = st.text_input("Enter Google API Key", type="password")
    elif llm_provider == "OpenRouter":
        openrouter_api_key = st.text_input("Enter OpenRouter API Key", type="password")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

generate_button = st.button("Generate MCQs from PDF")

if generate_button:
    if not uploaded_file:
        st.error("Please upload a PDF document.")
        st.stop()

    num_questions = num_questions_input
    llm = None
    config = None

    with st.spinner(f"Generating {num_questions} MCQs from the uploaded PDF using {llm_provider}..."):
        try:
            if llm_provider == "Gemini":
                if not gemini_api_key:
                    st.error("Please enter your Google API Key.")
                    st.stop()
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-exp-0827", google_api_key=gemini_api_key)
                config = LLMConfig(custom_model=llm)

            elif llm_provider == "OpenRouter":
                if not openrouter_api_key:
                    st.error("Please enter your OpenRouter API Key.")
                    st.stop()
                openrouter_model_name = "deepseek/deepseek-r1-distill-llama-70b"  # Or let user select in future
                openrouter_base_url = "https://openrouter.ai/api/v1"
                llm = ChatOpenAI(api_key=openrouter_api_key, model_name=openrouter_model_name, base_url=openrouter_base_url)
                config = LLMConfig(custom_model=llm)

            if config:
                client = Educhain(config)
                pdf_questions = client.qna_engine.generate_questions_from_data(
                    source=uploaded_file,
                    source_type="pdf",
                    num=num_questions
                )

                # Debugging: Inspect the response
                st.write("Debug - PDF Questions Response:", pdf_questions)

                # Handle empty or invalid responses
                if not pdf_questions or not hasattr(pdf_questions, "questions"):
                    st.error("No questions were generated. Please check the input PDF and try again.")
                    st.stop()

                questions = pdf_questions.questions
                if not questions:
                    st.error("No questions were generated. Please check the input PDF and try again.")
                    st.stop()

                st.success(f"Successfully generated {len(questions)} MCQs!")
                for q_data in questions:
                    st.markdown("---")
                    st.subheader(f"Question: {q_data.question}")

                    options_display = ""
                    for idx, option in enumerate(q_data.options, start=1):
                        options_display += f"{chr(64+idx)}. {option}  \n"
                    st.markdown(f"**Options:**  \n{options_display}")
                    st.markdown(f"**Correct Answer:** {q_data.answer}")
                    if q_data.explanation:
                        st.markdown(f"**Explanation:** {q_data.explanation}")

        except Exception as e:
            st.error(f"An error occurred during question generation: {e}")

st.markdown("---")
st.markdown("Powered by Educhain")
