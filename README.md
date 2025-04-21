# Real-Time Mask Detection with Webcam Integration

## Overview

This project implements a real-time mask detection system using a webcam interface. The system leverages a pre-trained deep learning model to classify whether a person is wearing a mask, has an incorrect mask, or is not wearing a mask at all. The detection system integrates face detection capabilities to ensure mask detection is performed only on human faces. Users can access the webcam stream for real-time predictions, as well as upload images for mask classification.

The model was trained on a dataset stored as a pickle file in the `data` folder. The model's architecture and training process are documented in the Jupyter notebook located in the `notebook` section.

## Features

* **Webcam Integration:** Users can access their webcam and get real-time mask detection results.
* **Face Detection:** The system detects faces in the webcam stream and performs mask detection only on the detected faces.
* **Mask Prediction:** The system categorizes the mask status into three classes:
    * No Mask
    * Mask
    * Incorrect Mask
* **Image Upload:** Users can upload an image for mask classification.
* **Interactive Interface:** The application provides buttons to start and stop webcam detection, along with visual feedback on predictions.

## Technologies Used

* **Flask:** Web framework to build the application.
* **PyTorch:** Deep learning framework for model inference.
* **OpenCV:** For face detection in webcam frames.
* **HTML/CSS:** Frontend for the user interface.
* **JavaScript:** Handles webcam interactions and AJAX requests.
* **PIL (Python Imaging Library):** For image processing.
* **Pickle:** To save and load the dataset for training.

## How It Works

1.  **Webcam Access:** The user clicks the 'Start Detection' button to access their webcam.
2.  **Mask Detection:** Once the webcam stream is active, the system captures frames from the webcam, processes them for face detection, and uses the pre-trained deep learning model to predict whether the person is wearing a mask, has an incorrect mask, or is not wearing a mask at all.
3.  **Real-Time Predictions:** The system provides real-time predictions on the webcam feed. A rectangle is drawn around the detected face, and the mask status is displayed.
4.  **Image Upload:** Users can upload an image, and the system classifies whether the person in the image is wearing a mask or not.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/real-time-mask-detection.git](https://github.com/your-username/real-time-mask-detection.git)
    cd real-time-mask-detection
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # On Windows, use:
    # venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```
5.  **Open the application:** Navigate to `http://127.0.0.1:5000` in your browser.

## File Structure

```bash
/real-time-mask-detection
│
├── app.py                  # Main Flask application
├── /app
│   ├── /static             # Static files (CSS, JS, images)
│   └── /templates          # HTML files
│
├── /model
│   └── mask_detector.pth   # Pre-trained model
│
├── /utils
│   └── mask_detector.py    # Mask detection logic
│
├── /webcam.js              # JavaScript for handling webcam interactions
├── /index.html             # Frontend UI
├── /notebook/face-mask-detection.ipynb  # Jupyter notebook for training the model
├── /data/masks_dataset.pickle # Pickle file containing images and labels for training
└── requirements.txt        # Python dependencies

## Model Details

The `MaskModel` is a pre-trained PyTorch model designed to classify whether a person is wearing a mask, has an incorrect mask, or is not wearing a mask. The model was trained on a custom dataset containing images of faces with labels indicating the mask status. The model architecture and training details are documented in the Jupyter notebook located in the `notebook` directory (`notebook/face-mask-detection.ipynb`).

### Training Process:

* The model was trained using a dataset stored as a pickle file (`masks_dataset.pickle`), which contains both images and their corresponding labels.
* The training process is described in the `notebook/face-mask-detection.ipynb` file, where the dataset is preprocessed, the model is trained, and the final model weights are saved in `model/mask_detector.pth`.

## Face Detection

Face detection is implemented using OpenCV's Haar Cascade Classifier. The system first detects faces within the webcam feed, and once a face is detected, the mask detection model is applied to classify the mask status.

## Future Enhancements

* **Multiple Face Detection:** Detect and classify multiple faces within the webcam frame.
* **Mobile Support:** Enhance the frontend for better mobile support.
* **Error Handling:** Improve error handling for edge cases like no faces detected or no webcam access.