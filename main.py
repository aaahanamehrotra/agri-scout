import os
import base64
from groq import Groq
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

def encode_image_to_base64(image_path: str) -> str:
    """Encodes a local image file to a base64 data URI string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Determine mime type (defaulting to jpeg/png check)
    ext = image_path.lower().split('.')[-1]
    mime_type = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded_string}"

def classify_plant_leaf_groq(image_path: str):
    """
    Classifies a plant leaf image as healthy or diseased using Groq's vision model.
    """
    # 1. Initialize the Groq client
    # The client automatically picks up the GROQ_API_KEY environment variable.
    client = Groq()

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # 2. Convert local image to base64 data URL format required for API payloads
    try:
        image_url = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 3. Construct the prompt
    prompt = """
    You are an expert plant pathologist and agricultural scientist. 
    Analyze this leaf image and provide a structured assessment:
    
    1. **Classification**: State clearly whether the plant is **Healthy** or **Diseased**.
    2. **Condition / Disease Name**: If diseased, name the specific disease, pest damage, or nutrient deficiency (e.g., Powdery Mildew, Blight, Nitrogen Deficiency). If healthy, state "None".
    3. **Confidence Level**: Estimate your confidence (Low, Medium, High).
    4. **Key Visual Symptoms**: Briefly bullet point the visual indicators observed (e.g., chlorosis, necrotic spots, fungal growth).
    5. **Recommended Action**: Provide brief treatment or care recommendations.
    """

    print("Analyzing leaf image with Groq...")

    try:
        # 4. Call Groq's chat completion endpoint with vision payload
        # Note: Replace the model name with the current Groq vision-supported model identifier
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b", # or your available multimodal groq model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
        )

        # 5. Print results
        print("\n" + "="*40)
        print("PLANT HEALTH DIAGNOSTIC REPORT (GROQ)")
        print("="*40)
        print(response.choices[0].message.content)
        print("="*40)

    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    # Replace with the path to your leaf image
    # sample_image_path = "data:image/jpeg;base64,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
    # "
    sample_image_path = "dataset-card.jpg"
    
    # Run the classification
    classify_plant_leaf_groq(sample_image_path)