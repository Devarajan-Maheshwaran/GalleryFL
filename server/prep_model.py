import tensorflow as tf
import os

def prep_feature_extractor():
    print("Preparing MobileNetV3-Small feature extractor...")
    os.makedirs("models", exist_ok=True)
    
    # Load MobileNetV3Small without the top classification layer
    # Output is the 1024-dimensional feature vector before the final dense layer
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg' # GlobalAveragePooling2D
    )
    
    # Freeze the base model (it will only be used for feature extraction)
    base_model.trainable = False
    
    # Convert to TFLite
    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    
    # Optional: Quantize the base model for faster inference and smaller size on device
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    output_path = "models/base_model.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Feature extractor saved to {output_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    prep_feature_extractor()
