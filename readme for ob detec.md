# 🔍 Object Detector (DETR + Gradio)

A simple web app that detects objects in an image using Meta's **DETR (DEtection TRansformer)** model and draws labeled bounding boxes around them. Built with Hugging Face Transformers and Gradio.

## Demo

Upload any image, and the app returns the same image with red bounding boxes and labels (object name + confidence score).

> Add a screenshot here: `![Demo](images/demo.png)`

## Features

- Detects 80+ common object classes (people, cars, animals, furniture, etc.)
- Uses the pre-trained `facebook/detr-resnet-50` model
- Draws bounding boxes with class labels and confidence scores
- Clean web UI with Gradio, no frontend code needed
- Shareable public link via `share=True`

## Tech Stack

- Python
- Hugging Face Transformers (object-detection pipeline)
- PyTorch
- Gradio
- Pillow (PIL)

## How It Works

1. The user uploads an image through the Gradio interface.
2. The image is passed to the Hugging Face `object-detection` pipeline using DETR.
3. The model returns detected objects with labels, confidence scores, and box coordinates.
4. Pillow draws the boxes and labels on a copy of the image.
5. The annotated image is displayed in the app.

## Project Structure

```
object-detector/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/your-username/object-detector.git
cd object-detector
pip install -r requirements.txt
```

`requirements.txt`:

```
gradio
transformers
torch
pillow
timm
```

### Run

```bash
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`). The first run downloads the model, so it may take a minute.

## Example Output

The terminal prints raw detections like:

```
[{'score': 0.99, 'label': 'person', 'box': {'xmin': 10, 'ymin': 20, 'xmax': 200, 'ymax': 300}}]
```

## Possible Improvements

- Add a confidence threshold slider
- Support video or webcam input
- Show a table of detected objects and counts
- Deploy on Hugging Face Spaces

## Acknowledgements

- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (Facebook AI Research)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio](https://www.gradio.app/)
- Project idea from the GenAILearniverse series

## Author

**Akash**
- LinkedIn: [your-profile-link]
- Email: your-email@example.com
