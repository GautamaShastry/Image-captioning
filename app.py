import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set web page title and layout
st.set_page_config(
    page_title="Image Caption Generator", 
    page_icon="📷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and tokenizer
@st.cache(allow_output_mutation=True)
def load_models_and_tokenizer():
    # Load MobileNetV2 for image feature extraction
    mobilenet = MobileNetV2(weights='imagenet')
    feature_extractor = Model(inputs=mobilenet.inputs, outputs=mobilenet.layers[-2].output)
    
    # Load the trained LSTM model
    model = tf.keras.models.load_model('model.h5')
    
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    return feature_extractor, model, tokenizer

# Load models and tokenizer
feature_extractor, lstm_model, tokenizer = load_models_and_tokenizer()

# Utility functions
def preprocess_image(image_file):
    """
    Preprocesses the uploaded image for MobileNetV2.
    """
    image = load_img(image_file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(image, model):
    """
    Extracts features from the image using MobileNetV2.
    """
    return model.predict(image, verbose=0)

def predict_caption(model, image_features, tokenizer, max_caption_length=34):
    """
    Generates a caption for an image using the trained LSTM model.
    """
    caption = ["startseq"]
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = tokenizer.index_word.get(predicted_index, None)
        
        if predicted_word is None or predicted_word == "endseq":
            break
        caption.append(predicted_word)
        
    return " ".join(caption[1:])

# Streamlit UI
st.title("📷 Image Caption Generator")
st.markdown("""
<div style="text-align: center; font-size: 1.1rem;">
Upload an image, and this app will generate a meaningful caption for it using AI.
</div>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("📋 Instructions")
st.sidebar.info("""
1. Upload an image file in JPG, JPEG, or PNG format.
2. Wait for the model to process the image.
3. View the generated caption on the screen.
""")

# Upload image
uploaded_image = st.file_uploader("Upload an Image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.markdown("<h3 style='text-align: center;'>Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("✨ Generating a caption for your image..."):
        # Preprocess image
        preprocessed_image = preprocess_image(uploaded_image)
        
        # Extract features
        image_features = extract_features(preprocessed_image, feature_extractor)
        
        # Generate caption
        generated_caption = predict_caption(lstm_model, image_features, tokenizer)

    # Display the generated caption
    st.markdown("<h3 style='text-align: center;'>Generated Caption</h3>", unsafe_allow_html=True)
    st.success(f"“{generated_caption}”")

# Footer
st.markdown("""
<hr>
<div style="text-align: center; font-size: 0.9rem;">
Developed with ❤️ by [Your Name](https://your-portfolio-link.com)
</div>
""", unsafe_allow_html=True)
