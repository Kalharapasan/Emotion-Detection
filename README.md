# Emotion Detection

Tkinter-based emotion detection app powered by TensorFlow, OpenCV, and a CNN model trained on FER-2013 style face images.

## Features

- Image upload with face detection and emotion prediction
- Live webcam detection with on-screen labels and session counts
- Local training workflow from a FER-2013 dataset folder
- Google Colab workflow for training in the cloud
- Model info panel with architecture and validation details

## Project Structure

- `app.py` - main desktop application
- `model/` - saved model and metadata
- `model_training/Emotion_Detection.ipynb` - training notebook
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- TensorFlow 2.21+
- OpenCV
- Pillow
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

If a model exists at `model/emotion_model.h5`, the app loads it automatically on startup.

## Training Locally

Use the **Train Model** tab if you have a FER-2013 style dataset with this layout:

```text
dataset/
	train/
		Angry/
		Disgust/
		Fear/
		Happy/
		Sad/
		Surprise/
		Neutral/
	test/
		Angry/
		Disgust/
		Fear/
		Happy/
		Sad/
		Surprise/
		Neutral/
```

The trained model is saved to `model/emotion_model.h5` and the metadata is written to `model/model_meta.json`.

## Notes

- The app suppresses TensorFlow startup noise from oneDNN and absl logging.
- If you update TensorFlow or Keras and the saved model stops loading, retrain the model in the same environment or refresh the saved weights.
