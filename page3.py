import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, resnet50
from PIL import Image, ImageOps, ExifTags
import numpy as np
import torch
import cv2

def run():
    st.markdown(
        """
        <link rel="icon" href="static/brickicon3.png" type="image/x-icon">
        <style>
            .reportview-container {
                background-color: #f7f9fc;
                padding-top: 20px;
            }
            .sidebar .sidebar-content {
                background-color: #f7f9fc;
            }
            .main-header {
                color: #ff6347;
                text-align: center;
            }
            .footer {
                text-align: center;
                padding: 10px;
                font-size: small;
                color: #666;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    imagelogo = Image.open("static/head3.png")
    st.image(imagelogo, use_column_width=True, width=600)

    # Add space below the logo
    st.write("")  # Creates a blank line
    st.write(" ")  # Creates an extra line for more space
    st.write(" ")  # Adjust the number of empty lines for desired spacing
    # Sidebar for app information
    with st.sidebar.expander("About the Version"):
        st.write("""
        This version of BrickSense App is a powerful tool designed to automatically detect cracks 
        in brick walls, leveraging cutting-edge deep learning technique. Built using a Convolutional 
        Neural Network (CNN) model pre-trained on a dataset of more than 150,000 images, the app 
        specializes in identifying structural defects in brick walls with high accuracy. The app 
        can analyze and classify a single image at a time, under three categoies namely noraml, cracked 
        and not a wall, making it easy for users to quickly check the condition of brick structures in real-time.
        """)
        st.write("""
        **Developed by:**  
        Talha Bin Tahir  
        **Email:** talhabtahir@gmail.com
        """)

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('170kmodelv9_version_cam_1.keras')

    @st.cache_resource
    def load_yolo_model():
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    @st.cache_resource
    def load_imagenet_model():
        return ResNet50(weights='imagenet')

    model = load_model()
    yolo_model = load_yolo_model()
    imagenet_model = load_imagenet_model()

    file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

    def correct_orientation(image):
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(orientation, 1)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        return image

    def import_and_predict(image_data, model):
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    def analyze_with_yolo(image_path):
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            st.error(f"Error: Could not open or find the image at {image_path}")
            return None
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)
        return results.pandas().xyxy[0]

    def import_and_predict_imagenet(image_data, model):
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32)
        img_reshape = np.expand_dims(img, axis=0)
        img_preprocessed = resnet50.preprocess_input(img_reshape)
        prediction = model.predict(img_preprocessed)
        return resnet50.decode_predictions(prediction, top=3)[0]

    if file is None:
        st.info("Please upload an image file to start the detection.")
    else:
        try:
            image = Image.open(file)
            image = correct_orientation(image)
            st.image(image, caption="Uploaded Image (Corrected Orientation)", use_column_width=True)
            image = image.convert("RGB")
            image_path = '/tmp/uploaded_image.jpg'
            image.save(image_path, format='JPEG')

            # Step 1: TensorFlow prediction
            predictions = import_and_predict(image, model)
            predicted_class = np.argmax(predictions[0])
            prediction_percentages = predictions[0] * 100

            st.write(f"**Prediction Percentages:**")
            st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
            st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
            st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")

            # Step 2: Proceed with YOLO and ImageNet if predicted as "Not a Wall"
            if predicted_class == 2:
                st.warning(f"⚠️ This is not a brick wall. Proceeding with YOLO and ImageNet detection.")

                yolo_results = analyze_with_yolo(image_path)
                if yolo_results is not None and not yolo_results.empty:
                    st.write("#### YOLOv5 Classification Results:")
                    st.write(f"YOLOv5 detected: {', '.join(yolo_results['name'].unique().tolist())}")

                imagenet_predictions = import_and_predict_imagenet(image, imagenet_model)
                if imagenet_predictions:
                    st.write("#### ImageNet Classification Results:")
                    for _, class_name, score in imagenet_predictions:
                        st.write(f"Class: {class_name}, Score: {score:.4f}")

            elif predicted_class == 0:
                st.success(f"✅ This is a normal brick wall.")
            elif predicted_class == 1:
                st.error(f"❌ This wall is a cracked brick wall.")

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")

    st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
