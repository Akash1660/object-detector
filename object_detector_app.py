import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

def draw_bounding_boxes(image, detections, font_path=None, font_size=20):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin = int(box['xmin'])
        ymin = int(box['ymin'])
        xmax = int(box['xmax'])
        ymax = int(box['ymax'])

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        text_bbox = draw.textbbox((xmin, ymin), text, font=font)

        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="black")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image

def detect_object(image):
    output = object_detector(image)
    print(output)
    return draw_bounding_boxes(image, output)

demo = gr.Interface(
    fn=detect_object,
    inputs=[gr.Image(label="Select Image", type="pil")],
    outputs=[gr.Image(label="Processed Image", type="pil")],
    title="@GenAILearniverse Project 6: Object Detector",
    description="Upload an image and this app will detect objects inside it using DETR."
)

demo.launch(share=True)
