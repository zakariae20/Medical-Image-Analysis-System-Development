import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Register the custom objects
tf.keras.utils.get_custom_objects().update({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

app = Flask(__name__)

# Absolute paths to your image and model files
image_path = r"C:\Users\HP\Documents\Image_Processing_Project\project\static\images\uploaded_image.png"
save_path = r"C:\Users\HP\Documents\Image_Processing_Project\project\static\images"
resnet_model_path = r"C:\Users\HP\Documents\Image_Processing_Project\project\ResNet50-model.keras"
unet_model_path = r"C:\Users\HP\Documents\Image_Processing_Project\project\U-net_model.keras"

# Check if the files exist
if not os.path.exists(resnet_model_path):
    raise FileNotFoundError(f"File not found: {resnet_model_path}")
if not os.path.exists(unet_model_path):
    raise FileNotFoundError(f"File not found: {unet_model_path}")

# Load the models with custom objects
resnet_model = tf.keras.models.load_model(resnet_model_path)
unet_model = tf.keras.models.load_model(unet_model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

scaler_subset = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    global image_path  # Make sure to use the global image_path defined above
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(save_path, 'uploaded_image.png')
            file.save(image_path)
            image = cv2.imread(image_path)
            if image is None:
                return "Failed to load image."

            # Resize the image to the expected input size of the model (128x128)
            image_resized = cv2.resize(image, (128, 128))
            
            # Convert the image to grayscale if the model expects single-channel input
            if unet_model.input_shape[-1] == 1:
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                image_resized = np.expand_dims(image_resized, axis=-1)
            else:
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize the image if required by your model (e.g., scaling pixel values)
            image_resized = image_resized / 255.0
            
            # Predict the mask
            mask = unet_model.predict(np.expand_dims(image_resized, axis=0))[0]
            
            # Postprocess the mask (thresholding)
            mask = (mask > 0.5).astype(np.uint8)

            # Save the mask
            base_name = os.path.basename(image_path)
            mask_name = os.path.splitext(base_name)[0] + "_mask.png"
            mask_path = os.path.join(save_path, mask_name)
            cv2.imwrite(mask_path, mask * 255)

            # Apply contours to the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = image_resized.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Red contours
            
            # Save the result with contours
            contour_image_path = os.path.join(save_path, 'uploaded_image_contour.png')
            cv2.imwrite(contour_image_path, (contour_image * 255).astype(np.uint8))

            # Process with ResNet-50
            categories = [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltrate",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "No finding"
            ]
            # categories = ["no finding", "finding"]  # Example categories, adjust as needed
            
            # Get the original size of the image
            original_image = load_img(image_path)
            original_size = original_image.size  # (width, height)

            # Preprocess the image for ResNet-50
            preprocessed_image = preprocess_image(image_path, (128, 128))

            # Predict using the ResNet-50 model
            class_predictions, bbox_prediction = predict(resnet_model, preprocessed_image)

            # Post-process the prediction
            confidence_threshold = 0.5  # Define your confidence threshold
            class_label, bbox = post_process_prediction(class_predictions, bbox_prediction, categories, original_size, (128, 128), confidence_threshold)

            # Combine contours and bounding box
            combined_image = draw_bbox_and_contours(image, contours, bbox, class_label)

            # Save the combined result
            combined_image_path = os.path.join(save_path, 'uploaded_image_combined.png')
            cv2.imwrite(combined_image_path, combined_image)
            save_dir = r'C:\Users\HP\Documents\Image_Processing_Project\project\static\images\uploaded_image_combined.png'

            # Get the original size of the image
            original_image = load_img(image_path)
            original_size = original_image.size  # (width, height)

            # Preprocess the image
            target_size = (128, 128)
            preprocessed_image = preprocess_image(image_path, target_size)

            # Predict using the model
            class_predictions, bbox_prediction = predict(resnet_model, preprocessed_image)

            

            # Post-process the prediction
            confidence_threshold = 0.5  # Define your confidence threshold
            class_label, bbox = post_process_prediction(class_predictions, bbox_prediction, categories, original_size, target_size, confidence_threshold)

            def display_and_save_results(image_path, class_label, bbox, save_dir):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

                if class_label != "no finding":
                    # Unpack the bounding box coordinates
                    x, y, width, height = bbox
                    x = int(x)
                    y = int(y)
                    width = int(width)
                    height = int(height)

                    # Draw the bounding box
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(image, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Save the image if save_path is provided
                save_path = os.path.join(save_dir, os.path.basename(image_path))
                # Convert the image back to BGR before saving
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path,image_bgr)
                            
            # Display and save the results
            display_and_save_results(image_path, class_label, bbox, save_dir)

            # Pass the results to the template
            return render_template('result.html', image_path=image_path, mask_path=mask_path, combined_image_path=combined_image_path, class_label=class_label)
    
    return render_template('result.html')


def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(model, image):
    predictions = model.predict(image)
    class_predictions = predictions[0]  # Assuming the first output is class
    bbox_prediction = predictions[1]  # Assuming the second output is bounding box
    return class_predictions, bbox_prediction

def post_process_prediction(class_predictions, bbox_prediction, categories, original_size, target_size, confidence_threshold=0.5):
    class_id = np.argmax(class_predictions, axis=-1)[0]
    confidence = class_predictions[0, class_id]
    
    if confidence < confidence_threshold:
        return "no finding", [0, 0, 0, 0]
    
    class_label = categories[class_id + 1]  # Adjust for zero-based indexing
    bbox = bbox_prediction[0]
    
    # Rescale bbox to the original image size
    orig_width, orig_height = original_size
    target_width, target_height = target_size
    x_scale = orig_width / target_width
    y_scale = orig_height / target_height
    
    bbox[0] *= x_scale
    bbox[1] *= y_scale
    bbox[2] *= x_scale
    bbox[3] *= y_scale
    
    return class_label, bbox

def draw_bbox_and_contours(image, contours, bbox, class_label):
    image_copy = image.copy()
    
    # Draw contours
    cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 1)  # Red contours

    # Draw bounding box if there is a finding
    if class_label != "no finding":
        x, y, width, height = bbox
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        # Draw the bounding box
        cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image_copy, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_copy

# single_image_path = r'C:\Users\HP\Downloads\Image_Processing_Project\project\static\images\uploaded_image.png'  # Update with the actual image path


if __name__ == '__main__':
    app.run(debug=True)
