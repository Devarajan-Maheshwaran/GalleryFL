import os
import csv
import argparse
import random
import shutil
from coco_to_fgt_mapping import COCO_TO_FGT

# Standard COCO 80 classes
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def check_dataset_status(data_dir):
    """Check what assets are present and return their status."""
    images_train_dir = os.path.join(data_dir, "images", "train2017")
    labels_train_dir = os.path.join(data_dir, "labels", "train2017")
    
    images_exist = os.path.isdir(images_train_dir)
    labels_exist = os.path.isdir(labels_train_dir)
    
    return {
        "images_exist": images_exist,
        "labels_exist": labels_exist,
        "images_train_dir": images_train_dir,
        "labels_train_dir": labels_train_dir
    }

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO-COCO dataset to FGT Bootstrap Format")
    parser.add_argument("--data-dir", type=str, default="../data/coco_minitrain_10k", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="../data/bootstrap_seed", help="Path to output bootstrap dir")
    parser.add_argument("--max-total-images", type=int, default=5000, help="Max images to include in bootstrap")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    status = check_dataset_status(data_dir)
    
    print("=== Dataset Discovery Report ===")
    print(f"Images Directory Found: {status['images_exist']} ({status['images_train_dir']})")
    print(f"Labels Directory Found: {status['labels_exist']} ({status['labels_train_dir']})")
    print("================================\n")
    
    if not status["labels_exist"] or not status["images_exist"]:
        print("CRITICAL ERROR: Dataset structure invalid or missing.")
        return
        
    print("Annotations found! Proceeding with mapping...")
    
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    
    label_files = os.listdir(status['labels_train_dir'])
    random.shuffle(label_files)
    
    # We will just use up to max-total-images
    label_files = label_files[:args.max_total_images]
    
    val_count = int(len(label_files) * args.val_split)
    val_files = label_files[:val_count]
    train_files = label_files[val_count:]
    
    def process_split(split_name, files):
        csv_path = os.path.join(out_dir, f"labels_{split_name}.csv")
        mapped_images = 0
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'tags'])
            
            for label_file in files:
                if not label_file.endswith('.txt'): continue
                
                base_name = label_file.replace('.txt', '')
                img_file = base_name + '.jpg'
                src_img_path = os.path.join(status['images_train_dir'], img_file)
                
                if not os.path.exists(src_img_path):
                    continue
                    
                label_path = os.path.join(status['labels_train_dir'], label_file)
                
                fgt_tags = set()
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        if class_id < len(COCO_CLASSES):
                            coco_name = COCO_CLASSES[class_id]
                            if coco_name in COCO_TO_FGT:
                                fgt_tags.add(COCO_TO_FGT[coco_name])
                
                if fgt_tags:
                    # Write to CSV with relative path to original image
                    rel_img_path = f"../coco_minitrain_10k/images/train2017/{img_file}"
                    tags_str = ";".join(fgt_tags)
                    writer.writerow([rel_img_path, tags_str])
                    mapped_images += 1
                    
        return mapped_images

    print("Processing training split...")
    train_mapped = process_split("train", train_files)
    print("Processing validation split...")
    val_mapped = process_split("val", val_files)
    
    print(f"\nConversion complete!")
    print(f"Mapped {train_mapped} training images and {val_mapped} validation images to FGT taxonomy.")
    print(f"Output saved to {out_dir}")

if __name__ == "__main__":
    main()
