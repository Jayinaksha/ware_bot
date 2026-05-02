import os
import argparse
import errno
from PIL import Image
import numpy as np
from keras.preprocessing.image import array_to_img
from models import modelsClass

# Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def deblur_image(image_path, output_path, model_weights_path):
    """
    Deblurs a single image using the DeepBlind model.
    """
    try:
        blurred_img = Image.open(image_path)
        blurred_np = (1./255)*np.array(blurred_img)
        
        width, height = blurred_img.size
        
        # Initialize model
        models = modelsClass(height, width)
        model = models.getDeepBlind()
        
        if not os.path.exists(model_weights_path):
            print(f"Error: Model weights not found at {model_weights_path}")
            return False

        model.load_weights(model_weights_path)
        
        # Predict
        x = np.reshape(blurred_np, [1, height, width, 3])
        prediction = model.predict(x, batch_size=1, verbose=0)
        prediction = prediction[0, :, :, :]
        
        # Save
        deblurred_img = array_to_img(prediction)
        deblurred_img.save(output_path)
        print(f"Deblurred image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error deblurring image: {e}")
        return False

def main():
    desc = "DeepBlind - Blind deblurring method."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output folder')
    parser.add_argument('-w', '--weights', type=str, default='DeepBlind.hdf5', help='Path to model weights')
    args = parser.parse_args()

    # Create output directory
    try:
        os.makedirs(args.output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Process input
    if os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in files:
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, fname)
            deblur_image(in_path, out_path, args.weights)
    else:
        fname = os.path.basename(args.input)
        out_path = os.path.join(args.output, fname)
        deblur_image(args.input, out_path, args.weights)

if __name__ == "__main__":
    main()