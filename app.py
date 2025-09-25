import base64
import io
import os
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from scipy.signal import convolve2d

app = Flask(__name__)

# --- New: Configuration for saving uploads ---
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create the folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# ---------------------------------------------

def max_pool_2x2(image_array):
    """Performs a 2x2 max pooling operation on a color image array."""
    h, w, c = image_array.shape
    new_h, new_w = h // 2, w // 2
    pooled_image = np.zeros((new_h, new_w, c))
    
    for i in range(new_h):
        for j in range(new_w):
            window = image_array[i*2:i*2+2, j*2:j*2+2]
            luminance = window[:, :, 0] * 0.299 + window[:, :, 1] * 0.587 + window[:, :, 2] * 0.114
            brightest_pixel_coords = np.unravel_index(np.argmax(luminance), luminance.shape)
            pooled_image[i, j] = window[brightest_pixel_coords]
            
    return pooled_image

def min_pool_2x2(image_array):
    """Performs a 2x2 min pooling operation on a color image array."""
    h, w, c = image_array.shape
    new_h, new_w = h // 2, w // 2
    pooled_image = np.zeros((new_h, new_w, c))
    
    for i in range(new_h):
        for j in range(new_w):
            window = image_array[i*2:i*2+2, j*2:j*2+2]
            luminance = window[:, :, 0] * 0.299 + window[:, :, 1] * 0.587 + window[:, :, 2] * 0.114
            darkest_pixel_coords = np.unravel_index(np.argmin(luminance), luminance.shape)
            pooled_image[i, j] = window[darkest_pixel_coords]
            
    return pooled_image

def process_image(image_data, kernel_type, pooling_type):
    """
    Performs convolution and pooling based on the selected types.
    """
    # --- 1. Image Processing ---
    pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    my_image = np.array(pil_image)
    img = my_image / 255.0

    # --- 2. Kernel Selection & Convolution ---
    kernels = {
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "laplacian": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    }
    kernel = kernels.get(kernel_type, kernels["sharpen"])

    crop_size = 128
    crop_h = min(crop_size, img.shape[0])
    crop_w = min(crop_size, img.shape[1])
    cropped_img_array = img[:crop_h, :crop_w, :]
    
    out_r = convolve2d(cropped_img_array[:, :, 0], kernel, mode="valid")
    out_g = convolve2d(cropped_img_array[:, :, 1], kernel, mode="valid")
    out_b = convolve2d(cropped_img_array[:, :, 2], kernel, mode="valid")

    conv_output_array = np.stack([out_r, out_g, out_b], axis=-1)
    
    # Normalize for edge detectors, clip for others
    if kernel_type in ["sobel_x", "sobel_y", "laplacian"]:
         if conv_output_array.max() > conv_output_array.min():
            conv_output_array = (conv_output_array - conv_output_array.min()) / (conv_output_array.max() - conv_output_array.min())
    else:
         conv_output_array = np.clip(conv_output_array, 0, 1)

    # --- 3. Pooling ---
    if pooling_type == 'min':
        pooled_array = min_pool_2x2(conv_output_array)
    else: # Default to max pooling
        pooled_array = max_pool_2x2(conv_output_array)

    # --- 4. Convert arrays to image bytes for transport ---
    def to_base64_png(image_array):
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    return {
        "input_image": to_base64_png(cropped_img_array),
        "sharpened_image": to_base64_png(conv_output_array),
        "pooled_image": to_base64_png(pooled_array)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data_url = data['image']
        kernel_type = data.get('kernel_type', 'sharpen')
        pooling_type = data.get('pooling_type', 'max')
        
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # --- New: Save the uploaded image ---
        try:
            image_to_save = Image.open(io.BytesIO(image_bytes))
            # Generate a unique filename and save the image as a PNG
            filename = f"{uuid.uuid4()}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_to_save.convert('RGB').save(filepath, 'PNG')
            print(f"Image saved to {filepath}")
        except Exception as save_error:
            # Log the error but don't stop the main process
            print(f"Error saving uploaded image: {save_error}")
        # ------------------------------------

        result = process_image(image_bytes, kernel_type, pooling_type)
        
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image on server"}), 500

if __name__ == '__main__':
    app.run(debug=True)
