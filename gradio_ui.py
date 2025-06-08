import gradio as gr
import requests
import numpy as np
import cv2

# Define the FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"

MODELS = {
    "2-class": {
        "endpoint": "/predict-both/",
        "classes": ["Helmet", "Without_Helmet"]
    },
    "multi-class": {
        "endpoint": "/predict-both-multiclass/",
        "classes": [
            "Helmet", "Mask", "Without_Helmet", "NO-Mask", "NO-Safety Vest",
            "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
        ]
    }
}

def predict_and_display(image_path, model_choice, selected_classes):
    # Log the image file
    print(f"Image file: {image_path}")
    
    # Read the image file as bytes
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    # Send the image to the FastAPI /predict-both/ endpoint
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    endpoint = MODELS[model_choice]["endpoint"]

    # For multi-class, send selected_classes as a form field
    if model_choice == "multi-class":
        data = {"selected_classes": ",".join(selected_classes)}
        response = requests.post(f"{FASTAPI_URL}{endpoint}", files=files, data=data)
    else:
        # For 2-class, just send the image (no class filtering needed)
        response = requests.post(f"{FASTAPI_URL}{endpoint}", files=files)

    if response.status_code == 200:
        result = response.json()
        # Log the response
        print(f"Response: {result}")
        # Display the full image with detections
        full_image_url = f"{FASTAPI_URL}{result['image_url']}"
        # Display the cropped images (if any)
        cropped_images = result.get("cropped_images", [])
        # Log the cropped image filenames
        print(f"Cropped image filenames: {cropped_images}")
        return full_image_url, cropped_images
    else:
        # Log the error
        print(f"Invalid response: {response.status_code}, {response.text}")
        # Create a blank image with an error message
        error_image = np.zeros((100, 500, 3), dtype=np.uint8)
        cv2.putText(error_image, "Error: Invalid response from server", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_image, []

with gr.Blocks() as demo:
    gr.Markdown("# Helmet & PPE Detection")
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="2-class",
            label="Select Model"
        )
        class_select = gr.CheckboxGroup(
            choices=MODELS["2-class"]["classes"],
            value=MODELS["2-class"]["classes"],
            label="Classes to Detect"
        )
    image_input = gr.Image(label="Upload Image", type="filepath")
    detect_btn = gr.Button("Detect")
    output_image = gr.Image(label="Detections")
    gallery = gr.Gallery(label="Cropped Images (Selected Classes)")

    def update_classes(selected_model):
        return gr.CheckboxGroup.update(
            choices=MODELS[selected_model]["classes"],
            value=MODELS[selected_model]["classes"]
        )

    model_choice.change(update_classes, inputs=model_choice, outputs=class_select)
    detect_btn.click(
        predict_and_display,
        inputs=[image_input, model_choice, class_select],
        outputs=[output_image, gallery]
    )

# Launch the Gradio app with share=True
demo.launch(share=True)
