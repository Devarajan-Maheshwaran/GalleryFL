import os
import argparse
import json
import numpy as np
import tensorflow as tf
from taxonomy_parser import TaxonomyParser, get_model_schema, validate_schema
import pandas as pd

def check_dataset_availability(data_dir: str = "../data/bootstrap_seed") -> bool:
    """Checks if the expected dataset structure exists."""
    expected_files = [
        os.path.join(data_dir, "labels_train.csv"),
        os.path.join(data_dir, "labels_val.csv")
    ]
    expected_dirs = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val")
    ]
    
    missing = []
    for f in expected_files:
        if not os.path.isfile(f):
            missing.append(f)
    for d in expected_dirs:
        if not os.path.isdir(d):
            missing.append(d)
            
    if missing:
        print("\n[!] DATASET MISSING OR INCOMPLETE")
        print("The following expected paths were not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease ensure the dataset is placed in the 'data/bootstrap_seed' directory.")
        print("Expected format:")
        print("  data/bootstrap_seed/")
        print("    train/                 (directory of images)")
        print("    val/                   (directory of images)")
        print("    labels_train.csv       (format: image_path,tags)")
        print("    labels_val.csv         (format: image_path,tags)")
        print("Example CSV row: train/img001.jpg,\"beach;water\"")
        return False
    return True

def build_full_model(num_classes: int):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    projection = tf.keras.layers.Dense(1024, activation='relu', name='backbone_projection')(pool)
    frozen_base = tf.keras.Model(inputs=base_model.input, outputs=projection)
    frozen_base.trainable = False
    
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(frozen_base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='dense_2')(x)
    
    full_model = tf.keras.Model(inputs=frozen_base.input, outputs=outputs)
    return frozen_base, full_model

def cmd_prepare(args):
    print("=== PREPARE MODE ===")
    
    parser = TaxonomyParser("taxonomy.json")
    num_classes = parser.num_classes
    print(f"Loaded taxonomy with {num_classes} leaf classes.")
    
    os.makedirs("models", exist_ok=True)
    
    frozen_base, full_model = build_full_model(num_classes)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(frozen_base)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = "models/base_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {tflite_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")
    
    # 2. Extract weights and save initial_head_weights.npz
    print("Extracting head weights...")
    dense_1 = full_model.get_layer('dense_1')
    dense_2 = full_model.get_layer('dense_2')
    
    w1, b1 = dense_1.get_weights()
    w2, b2 = dense_2.get_weights()
    
    # Validate against schema
    weights_list = [w1, b1, w2, b2]
    try:
        validate_schema(weights_list, num_classes)
        print("Schema validation passed.")
    except ValueError as e:
        print(f"Schema validation failed: {e}")
        return
        
    npz_path = "models/initial_head_weights.npz"
    np.savez(
        npz_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2
    )
    print(f"Saved {npz_path}")
    
    # 3. Export model_schema.json
    schema = get_model_schema(num_classes)
    schema_dict = {
        "num_classes": num_classes,
        "layers": [{"name": name, "shape": shape} for name, shape in schema],
        "taxonomy_version": 1
    }
    schema_path = "models/model_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema_dict, f, indent=2)
    print(f"Saved {schema_path}")
    
    print("Prepare mode completed successfully.")

def create_dataset(csv_path, data_dir, taxonomy_classes, batch_size=16, is_training=True):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [224, 224])
        # MobileNetV3 expects [-1, 1] or [0, 255]?
        # tf.keras.applications.MobileNetV3Small preprocess_input does this.
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        return image, label

    paths = []
    labels = []
    for _, row in df.iterrows():
        paths.append(os.path.join(data_dir, row['image_path']))
        
        # Multi-hot encoding
        tags = str(row['tags']).split(';') if pd.notna(row['tags']) else []
        label_vector = np.zeros(len(taxonomy_classes), dtype=np.float32)
        for tag in tags:
            tag = tag.strip()
            if tag in taxonomy_classes:
                label_vector[taxonomy_classes.index(tag)] = 1.0
        labels.append(label_vector)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def cmd_bootstrap_train(args):
    print("=== BOOTSTRAP TRAIN MODE ===")
    if not check_dataset_availability():
        return
        
    parser = TaxonomyParser("taxonomy.json")
    classes = parser.leaf_names
    num_classes = parser.num_classes
    
    print("Loading datasets...")
    train_ds = create_dataset("../data/bootstrap_seed/labels_train.csv", "../data/bootstrap_seed", classes, batch_size=args.batch_size, is_training=True)
    val_ds = create_dataset("../data/bootstrap_seed/labels_val.csv", "../data/bootstrap_seed", classes, batch_size=args.batch_size, is_training=False)
    
    if train_ds is None:
        print("Training CSV missing. Exiting.")
        return
        
    frozen_base, full_model = build_full_model(num_classes)
    
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True)]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')
    ]
    
    print("Starting training...")
    history = full_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save weights
    print("Exporting head weights...")
    dense_1 = full_model.get_layer('dense_1')
    dense_2 = full_model.get_layer('dense_2')
    
    w1, b1 = dense_1.get_weights()
    w2, b2 = dense_2.get_weights()
    
    weights_list = [w1, b1, w2, b2]
    try:
        validate_schema(weights_list, num_classes)
    except ValueError as e:
        print(f"Schema validation failed: {e}")
        return
        
    os.makedirs("models", exist_ok=True)
    npz_path = "models/initial_head_weights.npz"
    np.savez(npz_path, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"Saved best weights to {npz_path}")
    
    # Save metrics
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open("models/bootstrap_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")

def cmd_evaluate(args):
    print("=== EVALUATE MODE ===")
    if not check_dataset_availability():
        return
        
    parser = TaxonomyParser("taxonomy.json")
    classes = parser.leaf_names
    num_classes = parser.num_classes
    
    npz_path = "models/initial_head_weights.npz"
    if not os.path.exists(npz_path):
        print(f"{npz_path} not found. Run bootstrap_train first.")
        return
        
    val_ds = create_dataset("../data/bootstrap_seed/labels_val.csv", "../data/bootstrap_seed", classes, batch_size=32, is_training=False)
    if val_ds is None:
        print("Validation set missing.")
        return
        
    frozen_base, full_model = build_full_model(num_classes)
    
    # Load weights
    data = np.load(npz_path)
    full_model.get_layer('dense_1').set_weights([data['w1'], data['b1']])
    full_model.get_layer('dense_2').set_weights([data['w2'], data['b2']])
    
    full_model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Evaluating on validation set...")
    predictions = full_model.predict(val_ds)
    
    # We need to get true labels from the dataset to compute F1
    true_labels = []
    for _, y in val_ds:
        true_labels.append(y.numpy())
    true_labels = np.concatenate(true_labels, axis=0)
    
    # Apply a standard threshold
    threshold = 0.5
    pred_binary = (predictions > threshold).astype(int)
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    per_class_f1 = f1_score(true_labels, pred_binary, average=None)
    macro_f1 = f1_score(true_labels, pred_binary, average='macro')
    
    support = true_labels.sum(axis=0)
    
    report = {
        "macro_f1": float(macro_f1),
        "threshold": threshold,
        "per_class": {}
    }
    
    for i, cls_name in enumerate(classes):
        report["per_class"][cls_name] = {
            "f1": float(per_class_f1[i]),
            "support": int(support[i])
        }
        
    os.makedirs("output", exist_ok=True)
    with open("output/bootstrap_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Macro F1: {macro_f1:.4f}")
    print("Saved report to output/bootstrap_eval_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGT Model Preparation and Training Pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    parser_prep = subparsers.add_parser("prepare", help="Prepare base model and initial weights")
    
    parser_train = subparsers.add_parser("bootstrap_train", help="Run bootstrap training on local dataset")
    parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_train.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model against local validation set")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        cmd_prepare(args)
    elif args.mode == "bootstrap_train":
        cmd_bootstrap_train(args)
    elif args.mode == "evaluate":
        cmd_evaluate(args)
