import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json

# from google import genai
# from google.genai import types
import google.generativeai as genai
from ultralytics.utils.plotting import colors

# --- Initialize Google Gemini Client ---
# genai.Client(api_key="")  # Replace with your API key
# client = genai.Client(api_key="")

model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

genai.configure(api_key="AIzaSyAyXZy3TFg7B2Ai7G6raW9Cep-qzDXkJo8")

# --- Gemini Inference Function ---
import io
import base64
def inference(image_pil, prompt, temp=0.5):
    # Convert PIL image to base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Manually create the image block
    image_block = {
        "inline_data": {
            "mime_type": "image/png",
            "data": img_base64
        }
    }

    # Construct the contents payload
    contents = [
        {"role": "user", "parts": [prompt, image_block]}
    ]

    # Generate content
    response = model.generate_content(
        contents=contents,
        generation_config={"temperature": temp}
    )

    return response.text

# --- Annotator Class ---
class CustomAnnotator:
    def __init__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.img = image.copy()

    def box_label(self, box, label='', color=(255, 0, 0), font_scale=0.4, box_thickness=3, text_thickness=1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(self.img, (x1, y1), (x2, y2), color, box_thickness)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w, h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            cv2.rectangle(self.img, (x1, y1 - h - 4), (x1 + w + 2, y1), color, -1)
            cv2.putText(self.img, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    def result(self):
        return self.img

# --- Clean Gemini Output ---
def clean_results(results):
    text = results.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    return text

# --- Streamlit Interface ---
st.title("🏗️ Home Drawing Change Detector (Gemini + Streamlit)")

# uploaded_img = st.file_uploader("📤 Upload Combined Drawing (Left: Existing | Right: Proposed)", type=["png", "jpg", "jpeg"])
uploaded_img="img2.png"

if uploaded_img:
    combined = Image.open(uploaded_img).convert("RGB")

    # st.subheader("🖼 Uploaded Drawing")
    # st.image(combined, caption="Combined Drawing", use_column_width=True)

    # Prompt assumes image has both existing and proposed drawings
    prompt = """
    
you are expert home architect and familar with drawing the home drawings and analyze new designs.
there are 2 images of the site drawing (existing and proposed),
first find out what are the new changes are proposed, look and analyze it properly for all tiny changes, dont't miss any change.
then detect 2d bounding boxes around the changes are made in proposed images with comparision to the existing drawing. label it with the name of the change..
    """
    output_prompt = "Return just box_2d and labels and a description of the change as JSON. No extra text."
    full_prompt = prompt + output_prompt

    with st.spinner("🔎 Analyzing changes using Gemini..."):
        response_text = inference(combined, full_prompt)
        try:
            result_json = json.loads(clean_results(response_text))
        except Exception as e:
            st.error("⚠️ Gemini failed to return valid JSON.")
            st.code(response_text)
            st.stop()
    st.success("Analyzed Image")
    with st.spinner("🔎 Processign image..."):
        # Annotate combined image
        w, h = combined.size
        annotator = CustomAnnotator(combined)
        
        changes=[]
        for idx, item in enumerate(result_json):
            y1, x1, y2, x2 = item["box_2d"]
            label = item.get("label", f"Change {idx+1}")
            description = item.get("description", "")

            x1 = x1 / 1000 * w
            x2 = x2 / 1000 * w
            y1 = y1 / 1000 * h
            y2 = y2 / 1000 * h
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True), font_scale=0.4, box_thickness=4)

            changes.append(f"**🔹 {label}**: {description}")


        annotated = Image.fromarray(annotator.result())

    st.subheader("📊 Side-by-Side View: Original vs Annotated")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼 Original Drawing")
        st.image(combined, use_column_width=True)

    # with col2:
    st.markdown("### 📝 Annotated Drawing with Hover Tooltips")

    import streamlit.components.v1 as components

    # Convert PIL image to base64
    buffered = io.BytesIO()
    annotated.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    height, width = annotated.size

    # Generate tooltip divs
    tooltip_divs = ""
    for idx, item in enumerate(result_json):
        y1, x1, y2, x2 = item["box_2d"]
        label = item.get("label", f"Change {idx+1}")
        description = item.get("description", "")

        # Rescale
        x1 = x1 / 1000 * width
        x2 = x2 / 1000 * width
        y1 = y1 / 1000 * height
        y2 = y2 / 1000 * height
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        tooltip_divs += f"""
        <div class="tooltip-box" style="
            position:absolute;
            left:{x1}px;
            top:{y1}px;
            width:{x2 - x1}px;
            height:{y2 - y1}px;
            border:2px solid red;
            box-sizing:border-box;
        ">
            <span class="tooltip-text">{description}</span>
        </div>
        """

    # Final HTML with hover effect CSS
    html = f"""
    <style>
    .tooltip-box:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}

    .tooltip-text {{
        visibility: hidden;
        width: 200px;
        background-color: black;
        color: #fff;
        text-align: left;
        padding: 6px 8px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        top: 100%;
        left: 0;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    </style>

    <div style="position:relative; display:inline-block;">
        <img src="data:image/png;base64,{img_b64}" style="display:block;">
        {tooltip_divs}
    </div>
    """


    components.html(html, height=height + 100)



    
    st.subheader("🔧Key Changes")
    # Optional: Show description below or beside image
    for i in changes:
        st.write(i)