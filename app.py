import os
import base64
from io import BytesIO
from groq import Groq
from PIL import Image
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Page configuration for mobile browsers
st.set_page_config(page_title="Plant Leaf Health Classifier", page_icon="🌿", layout="centered")

@st.cache_resource
def get_groq_client():
    return Groq()

def classify_image_with_groq(image_bytes):
    """Encodes image bytes and calls Groq's multimodal vision model."""
    client = get_groq_client()
    
    # Convert image bytes to base64 data URL
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{encoded_string}"

    prompt = """
    You are an expert plant pathologist and agricultural scientist. 
    Analyze this leaf image and provide a structured assessment:
    
    1. **Classification**: State clearly whether the plant is **Healthy** or **Diseased**.
    2. **Condition / Disease Name**: If diseased, name the specific disease, pest damage, or nutrient deficiency. If healthy, state "None".
    3. **Confidence Level**: Estimate your confidence (Low, Medium, High).
    4. **Key Visual Symptoms**: Briefly bullet point the visual indicators observed.
    5. **Recommended Action**: Provide brief treatment or care recommendations.
    """

    try:
        response = client.chat.completions.create(
            # Using current Groq vision-supported multimodal model ID
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"

# UI Layout
st.title("🌿 Agri Scout")
st.write("Take a picture of a plant leaf using your phone camera to diagnose its health instantly.")

# Phone Camera Input widget (automatically opens phone camera hardware on mobile browsers)
camera_image = st.camera_input("Take a picture of the leaf")

if camera_image is not None:
    # Read image bytes from the camera stream
    bytes_data = camera_image.getvalue()
    
    # Display preview
    image = Image.open(BytesIO(bytes_data))
    st.image(image, caption="Captured Leaf")
    
    if st.button("Diagnose Leaf Health", type="primary"):
        with st.spinner("Analyzing leaf with Groq vision..."):
            report = classify_image_with_groq(bytes_data)
            
            st.markdown("---")
            st.subheader("Diagnostic Report")
            st.markdown(report)