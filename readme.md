# Brain Tumor Detection using Supervised Learning

This project aims to detect and classify brain tumors using a supervised learning model. The model is capable of identifying different types of brain tumors from MRI images and suggesting precautions and procedures for treatment.

## Project Structure

- `app.py`: The main Flask application file.
- `models/brainTumor-4category-b64e50-categorical-no-gpu.h5`: The trained model file.
- `requirements.txt`: Python environment dependencies.
- `templates/index.html`: The main HTML file for the web interface.

## Getting Started

### Prerequisites

Make sure you have the following installed on your system:

- Python 3.x
- Flask
- Keras
- TensorFlow
- OpenCV
- PIL (Pillow)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    python app.py
    ```

### Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Upload an MRI image file using the provided form.
3. The application will display the type of tumor detected, if any.

### Model Information

The model is trained to classify images into four categories:
- Glioma Tumor Present
- Meningioma Tumor Present
- No Tumor present
- Pituitary Tumor Present

### Functionality

- `get_classname(classname)`: Converts the model prediction output into a human-readable format.
- `getresult(img)`: Processes the image and gets the prediction from the model.

### Directory Structure

    brain-tumor-detection/
    ├── app.py
    ├── models/
    │ └── brainTumor-4category-b64e50-categorical-no-gpu.h5
    ├── requirements.txt
    └── templates/
    └── index.html