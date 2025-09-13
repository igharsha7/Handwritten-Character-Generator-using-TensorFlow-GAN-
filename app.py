import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Character Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">✍️ Handwritten Character Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Generate realistic handwritten digits and letters using AI</div>', unsafe_allow_html=True)

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model_and_mappings():
    """Load the generator model and character mappings"""
    try:
        # Load the generator model
        generator = tf.keras.models.load_model('handwritten_generator.h5')
        
        # Load character mappings
        with open('character_mappings.pkl', 'rb') as f:
            mappings_data = pickle.load(f)
        
        mapping = mappings_data['mapping']
        reverse_mapping = mappings_data['reverse_mapping']
        NUM_CLASSES = mappings_data['NUM_CLASSES']
        LATENT_DIM = mappings_data['LATENT_DIM']
        
        return generator, mapping, reverse_mapping, NUM_CLASSES, LATENT_DIM
    except Exception as e:
        st.error(f"Error loading model or mappings: {str(e)}")
        st.error("Please ensure the following files are in the same directory as the app:")
        st.error("- handwritten_generator.h5")
        st.error("- character_mappings.pkl")
        return None, None, None, None, None

# Load model and mappings
generator, mapping, reverse_mapping, NUM_CLASSES, LATENT_DIM = load_model_and_mappings()

if generator is None:
    st.stop()

def correct_emnist_image(image_tensor, target_size=(128, 128)):
    """
    Corrects the EMNIST image orientation and improves resolution while maintaining proper size
    """
    if len(image_tensor.shape) == 2:
        image_tensor = tf.expand_dims(image_tensor, axis=-1)
    
    # First flipping left-right to correct horizontal inversion
    image = tf.image.flip_left_right(image_tensor)
    
    # Then rotating 90 degrees clockwise
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_up_down(image)
    
    # Enhancing resolution using bicubic interpolation
    enhanced_image = tf.image.resize(image, target_size, 
                                   method=tf.image.ResizeMethod.BICUBIC)
    
    # Enhancing contrast slightly to make the character more defined
    enhanced_image = tf.image.adjust_contrast(enhanced_image, 1.2)
    
    # Clipping values to ensure they stay in [0, 1] range
    enhanced_image = tf.clip_by_value(enhanced_image, 0, 1)
    
    return tf.squeeze(enhanced_image)

def generate_character_image(model, char, num_examples=1):
    """Generate image for a single character"""
    if char not in reverse_mapping:
        return None, f"Character '{char}' not supported"
    
    label_idx = reverse_mapping[char]
    label_vector = tf.one_hot([label_idx] * num_examples, depth=NUM_CLASSES)
    noise = tf.random.normal([num_examples, LATENT_DIM])
    generated = model([noise, label_vector], training=False)
    generated = (generated + 1) / 2.0  # Rescale to [0,1]
    
    images = []
    for i in range(num_examples):
        corrected_image = correct_emnist_image(tf.squeeze(generated[i]))
        images.append(corrected_image.numpy())
    
    return images, None

def create_combined_sentence_image(images):
    """Combine multiple character images into a single sentence image"""
    if not images:
        return None
    
    # Convert to PIL Images and combine horizontally
    pil_images = []
    for img in images:
        # Convert numpy array to PIL
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        pil_images.append(pil_img)
    
    # Calculate total width and max height
    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)
    
    # Create combined image
    combined = Image.new('L', (total_width, max_height), color=255)
    
    x_offset = 0
    for img in pil_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined

def image_to_bytes(image_array, format='PNG'):
    """Convert numpy array to downloadable bytes"""
    # Convert to PIL Image
    img_uint8 = (image_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes.getvalue()

def pil_to_bytes(pil_img, format='PNG'):
    """Convert PIL image to downloadable bytes"""
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes.getvalue()

# Sidebar configuration
st.sidebar.title("🎛️ Generation Options")

# Mode selection
mode = st.sidebar.radio(
    "Choose Generation Mode",
    ["Single Character", "Sentence/Word"],
    help="Select whether to generate a single character or multiple characters"
)

# Main content area
if mode == "Single Character":
    st.header("🔤 Single Character Generation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input")
        
        # Character input
        char_input = st.text_input(
            "Enter a character:",
            max_chars=1,
            placeholder="e.g., A, 5, z",
            help="Enter any digit (0-9), uppercase letter (A-Z), or lowercase letter (a-z)"
        )
        
        # Number of examples
        num_examples = st.slider(
            "Number of variations:",
            min_value=1,
            max_value=6,
            value=3,
            help="Generate multiple variations of the same character"
        )
        
        # Generate button
        generate_btn = st.button("🎨 Generate Character", type="primary")
        
        # Show available characters
        with st.expander("📋 Available Characters"):
            st.write("**Digits:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
            st.write("**Uppercase:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z")
            st.write("**Lowercase:** a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z")
    
    with col2:
        st.subheader("Generated Images")
        
        if generate_btn and char_input:
            with st.spinner("Generating handwritten character..."):
                images, error = generate_character_image(generator, char_input, num_examples)
                
                if error:
                    st.error(error)
                else:
                    # Display generated images
                    cols = st.columns(min(num_examples, 3))
                    for i, img in enumerate(images):
                        with cols[i % 3]:
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(img, cmap='gray')
                            ax.set_title(f"'{char_input}' - Variation {i+1}", fontsize=12)
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                    
                    # Store images in session state for download
                    st.session_state.generated_images = images
                    st.session_state.generated_char = char_input
        
        elif not char_input and generate_btn:
            st.warning("Please enter a character first!")
        elif not generate_btn:
            st.info("👆 Enter a character and click 'Generate Character' to see the magic!")
            
        # Download section for single character
        if hasattr(st.session_state, 'generated_images') and st.session_state.generated_images:
            st.subheader("💾 Download Options")
            download_format = st.selectbox("Choose format:", ["PNG", "JPEG"])
            
            if st.button("📥 Download All Variations"):
                # Create download for all variations
                if hasattr(st.session_state, 'generated_images'):
                    import zipfile
                    import tempfile
                    
                    # Create a temporary zip file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                            for i, img in enumerate(st.session_state.generated_images):
                                img_bytes = image_to_bytes(img, download_format)
                                filename = f"{st.session_state.generated_char}_variation_{i+1}.{download_format.lower()}"
                                zipf.writestr(filename, img_bytes)
                        
                        # Read the zip file
                        with open(tmp_file.name, 'rb') as f:
                            zip_bytes = f.read()
                        
                        st.download_button(
                            label=f"💾 Download ZIP ({len(st.session_state.generated_images)} images)",
                            data=zip_bytes,
                            file_name=f"character_{st.session_state.generated_char}_variations.zip",
                            mime="application/zip"
                        )

else:  # Sentence/Word mode
    st.header("📝 Sentence/Word Generation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input")
        
        # Sentence input
        sentence_input = st.text_input(
            "Enter text:",
            placeholder="e.g., Hello123, AI2024",
            help="Enter any combination of digits, uppercase and lowercase letters"
        )
        
        # Generation options
        st.subheader("Options")
        
        regenerate_btn = st.button("🔄 Generate New Variations", help="Generate new variations with different randomness")
        
        # Show character count and validation
        if sentence_input:
            valid_chars = [char for char in sentence_input if not char.isspace() and char in reverse_mapping]
            invalid_chars = [char for char in sentence_input if not char.isspace() and char not in reverse_mapping]
            
            st.write(f"**Valid characters:** {len(valid_chars)}")
            if invalid_chars:
                st.warning(f"**Unsupported characters:** {', '.join(set(invalid_chars))}")
    
    with col2:
        st.subheader("Generated Sentence")
        
        if sentence_input and (regenerate_btn or st.button("🎨 Generate Sentence", type="primary")):
            # Filter valid characters
            valid_chars = [char for char in sentence_input if not char.isspace() and char in reverse_mapping]
            
            if not valid_chars:
                st.error("No valid characters found! Please enter digits (0-9) or letters (A-Z, a-z).")
            else:
                with st.spinner("Generating handwritten text..."):
                    sentence_images = []
                    sentence_chars = []
                    
                    for char in valid_chars:
                        images, error = generate_character_image(generator, char, 1)
                        if not error:
                            sentence_images.append(images[0])
                            sentence_chars.append(char)
                    
                    if sentence_images:
                        # Display the sentence as a horizontal strip
                        fig, axes = plt.subplots(1, len(sentence_images), figsize=(2*len(sentence_images), 4))
                        
                        if len(sentence_images) == 1:
                            axes = [axes]
                        
                        for idx, (char, img) in enumerate(zip(sentence_chars, sentence_images)):
                            axes[idx].imshow(img, cmap='gray')
                            axes[idx].set_title(f"'{char}'", fontsize=12)
                            axes[idx].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Store in session state
                        st.session_state.sentence_images = sentence_images
                        st.session_state.sentence_chars = sentence_chars
                        st.session_state.original_sentence = sentence_input
                        
                        # Combined sentence image
                        st.subheader("📋 Combined Image")
                        combined_img = create_combined_sentence_image(sentence_images)
                        if combined_img is not None:
                            st.image(combined_img, caption=f"Generated: '{sentence_input}'", use_column_width=True)
        
        elif not sentence_input:
            st.info("👆 Enter some text and click 'Generate Sentence' to create handwritten text!")
        
        # Download section for sentences
        if hasattr(st.session_state, 'sentence_images') and st.session_state.sentence_images:
            st.subheader("💾 Download Options")
            download_format = st.selectbox("Choose format:", ["PNG", "JPEG"], key="sentence_format")
            
            col_download1, col_download2 = st.columns(2)
            with col_download1:
                if st.button("📥 Download Individual Images"):
                    if hasattr(st.session_state, 'sentence_images'):
                        import zipfile
                        import tempfile
                        
                        # Create a temporary zip file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                                for i, (char, img) in enumerate(zip(st.session_state.sentence_chars, st.session_state.sentence_images)):
                                    img_bytes = image_to_bytes(img, download_format)
                                    filename = f"{char}_{i+1}.{download_format.lower()}"
                                    zipf.writestr(filename, img_bytes)
                            
                            # Read the zip file
                            with open(tmp_file.name, 'rb') as f:
                                zip_bytes = f.read()
                            
                            st.download_button(
                                label=f"💾 Download ZIP ({len(st.session_state.sentence_images)} images)",
                                data=zip_bytes,
                                file_name=f"sentence_{st.session_state.original_sentence.replace(' ', '_')}_individual.zip",
                                mime="application/zip",
                                key="download_individual"
                            )
            with col_download2:
                if st.button("📥 Download Combined Image"):
                    if hasattr(st.session_state, 'sentence_images'):
                        combined_img = create_combined_sentence_image(st.session_state.sentence_images)
                        if combined_img:
                            img_bytes = pil_to_bytes(combined_img, download_format)
                            st.download_button(
                                label=f"💾 Download Combined {download_format}",
                                data=img_bytes,
                                file_name=f"sentence_{st.session_state.original_sentence.replace(' ', '_')}_combined.{download_format.lower()}",
                                mime=f"image/{download_format.lower()}",
                                key="download_combined"
                            )

# Footer
st.markdown("---")
st.markdown("""
### 📚 About This App

This handwritten character generator uses a **Generative Adversarial Network (GAN)** trained on the EMNIST dataset to create realistic handwritten characters.

**Features:**
- Generate individual characters (digits 0-9, letters A-Z, a-z)
- Create handwritten words and sentences
- Download high-quality images in PNG or JPEG format
- Multiple variations for creative flexibility

**Technical Details:**
- Model: Conditional GAN with 100-dimensional latent space
- Dataset: EMNIST By-Class (62 character classes)
- Image Resolution: Enhanced to 128x128 pixels
- Orientation: Automatically corrected for proper character display

**Supported Characters:** 0-9, A-Z, a-z (Total: 62 characters)

Created with ❤️ using TensorFlow and Streamlit
""")

# Additional sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Tips
- Try different characters to see the AI's interpretation
- Generate multiple variations to see different handwriting styles
- Combine letters and numbers for realistic text
- Download images for use in your projects

### 🔧 Technical Info
- **Model Type:** Conditional GAN
- **Training Dataset:** EMNIST By-Class
- **Latent Dimensions:** 100
- **Output Resolution:** 128x128 pixels
""")

if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()