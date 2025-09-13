# Handwritten Character Generator App

A Streamlit web application that generates realistic handwritten characters using a trained Generative Adversarial Network (GAN).

## Features

- **Single Character Generation**: Generate individual digits (0-9) and letters (A-Z, a-z) with multiple variations
- **Sentence/Word Generation**: Create handwritten words and sentences by combining multiple characters
- **High-Quality Output**: Images are enhanced to 128x128 resolution with proper orientation correction
- **Download Options**: Save generated images as PNG or JPEG, individually or as combined images
- **Interactive Interface**: User-friendly web interface with real-time generation

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files Are Present**:
   Make sure the following files are in the same directory as `app.py`:
   - `handwritten_generator.h5` (the trained generator model)
   - `character_mappings.pkl` (character mappings and model parameters)

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Open Your Browser**:
   The app will automatically open in your default browser, typically at `http://localhost:8501`

3. **Generate Characters**:
   - **Single Character Mode**: Enter any digit (0-9) or letter (A-Z, a-z) to generate handwritten variations
   - **Sentence Mode**: Enter any text to generate a complete handwritten sentence

4. **Download Images**:
   - Choose between PNG or JPEG format
   - Download individual character images as a ZIP file
   - Download combined sentence images

## Supported Characters

- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Uppercase Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Lowercase Letters**: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

**Total**: 62 supported characters

## Technical Details

- **Model Architecture**: Conditional GAN with 100-dimensional latent space
- **Training Dataset**: EMNIST By-Class dataset
- **Output Resolution**: 128x128 pixels (enhanced from original 28x28)
- **Frameworks**: TensorFlow for model inference, Streamlit for web interface

## File Structure

```
handwritten_generator_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── handwritten_generator.h5        # Trained generator model
├── character_mappings.pkl          # Character mappings and parameters
├── handwritten_discriminator.h5    # Discriminator model (optional)
└── training_history.pkl           # Training history (optional)
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check that TensorFlow is compatible with your system

2. **Model Loading Errors**:
   - Verify that `handwritten_generator.h5` and `character_mappings.pkl` are in the same directory as `app.py`
   - Check file permissions and ensure files are not corrupted

3. **Performance Issues**:
   - Generation may be slower on CPU-only systems
   - For faster inference, ensure TensorFlow can access GPU if available

4. **Character Not Supported**:
   - Only alphanumeric characters are supported (0-9, A-Z, a-z)
   - Special characters and symbols are not available

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: At least 4GB recommended
- **Storage**: ~500MB for model files and dependencies
- **GPU**: Optional, but recommended for faster generation

## License

This project is for educational and demonstration purposes. The EMNIST dataset is publicly available for research use.

## Contributing

Feel free to submit issues, feature requests, or improvements to enhance the application!