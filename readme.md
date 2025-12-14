


# Violence Detection System using MoBiLSTM

## 1\. About This Project

This project is a Deep Learning application designed to detect violent activities in real-time videos. It utilizes a hybrid architecture combining **Convolutional Neural Networks (CNN)** for feature extraction and **Recurrent Neural Networks (RNN)** for temporal sequence analysis.

### How It Works

The system employs a **MoBiLSTM** architecture:

  * **MobileNetV2:** Used as a pre-trained feature extractor to analyze individual frames and identify visual features.
  * **Bidirectional LSTM (BiLSTM):** Processes the sequence of features extracted by MobileNetV2 to understand the temporal context and motion patterns, distinguishing between violent and non-violent actions.

### Key Features

  * **Interactive Web Interface:** Built with **Gradio**, allowing users to easily upload models and videos for analysis.
  * **Frame-by-Frame Analysis:** breaks down videos into sequences (16 frames) to provide granular detection.
  * **Real-time Annotation:** Generates a processed video overlaying prediction labels (Violence/Non-Violence), bounding boxes, and confidence scores.
  * **Data Visualization:** Provides confidence trend graphs and probability charts to visualize how the model's certainty changes throughout the video.
  * **Reporting:** Exports a detailed CSV report of the analysis containing timestamps and prediction data.

-----

## 2\. How to Run

Follow these steps to set up the environment, train the model (if necessary), and launch the application.
Ensure you have Python installed (tested on Python 3.10+).

### Prerequisites

### step 1: Make a new virtual enviorment

Run the following command to make a virtual enviorment

```bash
python -m venv venv
```

Run the following command to activate the enviorment

```bash
venv\Scripts\activate
```


### Step 2: Install Dependencies

Run the following command to install all necessary libraries:

```bash
pip install -r require.txt
```

### Step 3: Obtain the Model

The application requires a trained Keras model (`.keras`) to function.

**Option A: Train a new model**
If you do not have a model file, you must generate one using the provided training script.

1.  Open `violence_detection_in_real_time_videos(4).py`.
2.  Ensure you have the Kaggle datasets configured or download the "Real Life Violence Situations" dataset manually.
3.  Run the script. Upon completion, it will save a file named `MoBiLSTM_model.keras`.

**Option B: Use an existing model**
If you already have `MoBiLSTM_model.keras`, skip to Step 3.

### Step 3: Launch the Application

Run the Gradio interface using the `violence.py` script:

```bash
python violence.py
```

### Step 4: Using the Interface

1.  Once the script runs, a local URL (e.g., `http://127.0.0.1:7860`) will appear in your terminal. Open this in your web browser.
2.  **Upload Model:** In the "Step 1" section, upload your `MoBiLSTM_model.keras` file and click **"Load Model"**. Wait for the "Model loaded successfully" message.
3.  **Upload Video:** In the "Step 2" section, upload the video file (MP4, AVI, etc.) you wish to analyze.
4.  **Analyze:** Click **"Analyze Video"**.
5.  **View Results:**
      * Watch the **Annotated Video** to see detections in real-time.
      * Check the **Summary** for the overall verdict.
      * Download the **CSV** for detailed timestamp data.