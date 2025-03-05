import streamlit as st
from pillow import Image
import requests
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_chat = ChatGroq(
    temperature=0.3,
    model_name="llama-3.3-70b-specdec",
    groq_api_key=GROQ_API_KEY
)

# Free APIs configuration
HF_API_URL = st.secrets["HF_API_URL"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Hybrid medical model configuration
MEDICAL_MODEL = st.secrets["MEDICAL_MODEL"]
GENERAL_MODEL = st.secrets["GENERAL_MODEL"]
PRODUCT_MODEL = st.secrets["PRODUCT_MODEL"]

def analyze_image(image_bytes, model_name):
    """Analyze image using free Hugging Face models"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(
            HF_API_URL + model_name,
            headers=headers,
            data=image_bytes,
            timeout=30
        )
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        return f"Analysis error: {str(e)}"

def get_medical_response(tags, query):
    """Hybrid medical analysis using CLIP tags + LLM"""
    prompt = ChatPromptTemplate.from_template("""
    As a medical assistant, analyze these image tags: {tags}
    For this question: {query}

    Provide:
    1. 3 possible conditions matching these symptoms
    2. Recommended diagnostic tests
    3. Urgency level (Emergency/Urgent/Routine)
    4. Clear disclaimer
    
    Format: Concise bullet points in plain text
    """)
    
    return (
        prompt | groq_chat | StrOutputParser()
    ).invoke({"tags": tags, "query": query})

def get_product_response(analysis, query):
    """Product analysis with safety checks"""
    prompt = ChatPromptTemplate.from_template("""
    Analyze product features: {analysis}
    For query: {query}

    Provide:
    1. Product identification
    2. Price estimate range (USD)
    3. 3 fictional purchase options
    4. Alternative suggestions
    
    Format: Simple text with line breaks
    """)
    
    return (
        prompt | groq_chat | StrOutputParser()
    ).invoke({"analysis": analysis, "query": query})

def get_general_response(analysis, query):
    """General image analysis"""
    prompt = ChatPromptTemplate.from_template("""
    Analyze image description: {analysis}
    For question: {query}

    Provide:
    1. Direct answer
    2. 3 relevant facts
    3. Related information
    
    Format: Short paragraphs in plain text
    """)
    
    return (
        prompt | groq_chat | StrOutputParser()
    ).invoke({"analysis": analysis, "query": query})

# Streamlit UI
st.set_page_config(page_title="AI Vision Assistant", layout="wide")
st.title("🖼️ AI Vision Assistant")
st.subheader("Upload Image + Select Analysis Type")

# Category selection
category = st.selectbox("Select Analysis Type:", 
                       ["General", "Medical", "Product"])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", 
                                type=["jpg", "jpeg", "png"])
user_query = st.text_input("Ask about the image:")

if uploaded_file and user_query:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        try:
            if category == "Medical":
                # Hybrid medical analysis
                raw_tags = analyze_image(image_bytes, MEDICAL_MODEL)
                tags = ", ".join([item["label"] for item in raw_tags]) if isinstance(raw_tags, list) else raw_tags
                response = get_medical_response(tags, user_query)
                disclaimer = "⚠️ This is not medical advice - Consult a doctor for diagnosis"
                
            elif category == "Product":
                analysis = analyze_image(image_bytes, PRODUCT_MODEL)
                response = get_product_response(str(analysis), user_query)
                disclaimer = "ℹ️ Price estimates are approximate"
                
            else:
                analysis = analyze_image(image_bytes, GENERAL_MODEL)
                response = get_general_response(str(analysis), user_query)
                disclaimer = ""

            st.subheader("Analysis Results")
            st.write(response)
            
            if disclaimer:
                st.warning(disclaimer)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Sidebar information
# Sidebar information
st.sidebar.markdown("## Category Guide")
st.sidebar.markdown("""
- **General**: Landscapes, objects, animals
- **Medical**: Skin conditions, X-rays, scans
- **Product**: Consumer goods, electronics, clothing
""")

st.sidebar.markdown("## How It Works")
st.sidebar.markdown("""
1. Select image type
2. Upload image
3. Ask your question
4. Get AI-powered analysis
""")
