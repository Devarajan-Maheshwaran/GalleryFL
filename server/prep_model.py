import tensorflow as tf
import os

def prep_feature_extractor():
    os.makedirs("models", exist_ok=True)

    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path = "models/base_model.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved {output_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    prep_feature_extractor()
