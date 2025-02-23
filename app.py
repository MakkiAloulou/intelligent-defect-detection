from PIL import ImageFont
from typing import Counter
import streamlit as st
from streamlit_option_menu import option_menu
from inference_sdk import InferenceHTTPClient
import tempfile
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Roboflow API Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="tfdqnOBSXkhNxZVfysgD"
)

# Store uploaded image and results in session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# Navigation Menu
selected_option = option_menu(
    None, ["Home", "Upload", "Results", 'Contact'], 
    icons=['house', 'cloud-arrow-up', 'clipboard-check', 'envelope'], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal"
)

# Home Page
if selected_option == "Home":
    st.title("🔍 Intelligent Defect Detection")

    st.markdown("""
    Welcome to our **AI-powered** solution for textile label inspection.  
    This system uses a model to automatically detect defects, 
    such as printing errors, stains, or structural defects.  

    ### 🎯 **Main Features:**
    - 📸 **Upload an image** and get a **real-time analysis**.
    - 🛠️ **Automatic classification** of detected defects.
    - 📊 **Interactive visualization** of results and statistics.
    - 🚀 **Intuitive and easy-to-use** interface.

    🔹 **Start now by uploading an image!**  
    """)

    if st.button("📂 Analyze an Image"):
        if st.session_state.analysis_done:
            st.session_state.analysis_done = False
        if st.session_state.uploaded_image:
            st.session_state.uploaded_image = None
        st.session_state.selected_page = "Upload"
        st.rerun()

# Upload Page
elif selected_option == "Upload":
    if st.session_state.analysis_done:
        st.session_state.analysis_done = False
    if st.session_state.uploaded_image:
        st.session_state.uploaded_image = None
    st.title("📂 Upload an Image")
    
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file

        # Prevent multiple analyses
        if not st.session_state.analysis_done:
            # Save the image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()

            # Start the analysis
            with st.spinner("🔍 Analyzing..."):
                result = CLIENT.infer(temp_file.name, model_id="intelligent-defect-detection/2")

            # Store the results and prevent re-analysis
            st.session_state.inference_result = result
            st.session_state.analysis_done = True  
            st.success("✅ Analysis completed! Access the results.")
            

# Results Page

elif selected_option == "Results":
    st.title("📊 Analysis Results")

    if st.session_state.uploaded_image and st.session_state.inference_result:
        # Show original image
        image = Image.open(st.session_state.uploaded_image)
        st.write("---")
        st.write("### 🖼️ Original Image")
        st.image(image, use_container_width=True)

        # Draw bounding boxes and add row indices inside the box
        draw = ImageDraw.Draw(image)

        # Try to load a larger font (if available, otherwise it will fall back to default)
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Adjust the font size as needed
        except IOError:
            font = ImageFont.load_default()  # Fallback to default if custom font is not available

        for idx, prediction in enumerate(st.session_state.inference_result['predictions']):
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            # Draw the bounding box
            draw.rectangle([(x - w/2, y - h/2), (x + w/2, y + h/2)], outline="green", width=3)
            
            # Calculate the position of the index text inside the box
            text_position = (x - w/2 + 5, y - h/2 + 5)  # Adjust position inside the box
            # Add the index as text inside the box
            draw.text(text_position, str(idx), font=font, fill="green")  # Set the font size

        st.write("---")
        st.write("### 🛠️ Detected Defects")
        st.image(image, use_container_width=True)
        st.write("---")
        inference_result = st.session_state.inference_result

        # Extract relevant data for the table from the inference result
        predictions = inference_result["predictions"]
        table_data = []

        # Loop through each prediction and extract the relevant details
        for idx, prediction in enumerate(predictions):
            table_data.append({
                "Detection ID": prediction["detection_id"],
                "Class": prediction["class"],
                "Confidence": prediction["confidence"],
                "Bounding Box (x, y, width, height)": f"({prediction['x']}, {prediction['y']}, {prediction['width']}, {prediction['height']})",
            })

        # Create a DataFrame to display the predictions in a table format
        df = pd.DataFrame(table_data)

        # Display the table in Streamlit
        st.write("### 🛠️ Defects Detected")
        st.dataframe(df)
        
        st.write("---")
        # Display raw inference results
        classes = [pred["class"] for pred in st.session_state.inference_result["predictions"]]
        class_counts = Counter(classes)

        # Create a bar chart
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(list(class_counts.keys()), list(class_counts.values()), color='skyblue')

        # Add labels and title
        ax.set_xlabel('Number of Occurrences')
        ax.set_title('Occurrences of Defect Classes')

        # Display the chart in the Streamlit app
        st.subheader("📝 Occurrences of Defect Classes")
        st.pyplot(fig)

    else:
        st.warning("❌ No results available. Please upload an image first.")


# Contact Page
elif selected_option == "Contact":
    st.title("📞 Contact Us")
    st.write("For any inquiries or support, feel free to contact us via email or phone.")
    st.write("---")
    coordinates = {
        "Makki Aloulou": {"email": "makkialoulou2005@gmail.com", "phone": "28716169"},
        "Mohamed Kallel": {"email": "kallelmohamed094@gmail.com", "phone": "21300465"},
        "Yessin Kolsi": {"email": "yassinex1928@gmail.com", "phone": "55497772"},
        "Ahmed Omar Ben Hazem": {"email": "benhazem.ahmedomar@gmail.com", "phone": "53724545"}
    }

    for name, contact in coordinates.items():
        st.subheader(name)
        st.write(f"📧 **Email:** {contact['email']}")
        st.write(f"📞 **Phone:** {contact['phone']}")
        st.write("---")
