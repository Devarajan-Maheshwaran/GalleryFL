import json
import os
from typing import Dict, List

# Standard COCO 80 classes mapping to FGT taxonomy
# FGT Taxonomy: 34 classes
# People: selfie, group_photo, portrait, crowd
# Places: beach, mountain, city, indoor, rural
# Activities: sports, cooking, celebration, work, travel
# Objects: food, vehicle, gadget, clothing, art
# Documents: screenshot, receipt, id_card, handwritten, printed
# Nature: landscape, animal, plant, sky, water
# Events: wedding, birthday, concert, graduation, holiday

COCO_TO_FGT: Dict[str, str] = {
    # People
    "person": "portrait", # Can't distinguish easily without faces, map to portrait as fallback
    
    # Vehicles -> Objects/Vehicle
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    
    # Animals -> Nature/Animal
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    
    # Gadgets -> Objects/Gadget
    "cell phone": "gadget",
    "tv": "gadget",
    "laptop": "gadget",
    "mouse": "gadget",
    "remote": "gadget",
    "keyboard": "gadget",
    "microwave": "gadget",
    
    # Objects/Food
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    
    # Objects/Clothing
    "tie": "clothing",
    "backpack": "clothing",
    "umbrella": "clothing",
    "handbag": "clothing",
    
    # Indoor/Furniture -> Places/Indoor
    "chair": "indoor",
    "couch": "indoor",
    "potted plant": "indoor", # Or Nature/Plant, let's go with Nature/Plant
    "bed": "indoor",
    "dining table": "indoor",
    "toilet": "indoor",
    "book": "indoor",
    "clock": "indoor",
    "vase": "indoor",
    "refrigerator": "indoor",
    "oven": "indoor",
    "toaster": "indoor",
    "sink": "indoor",
    
    # Nature/Plant
    "potted plant": "plant",
    
    # Activities/Sports
    "frisbee": "sports",
    "skis": "sports",
    "snowboard": "sports",
    "sports ball": "sports",
    "kite": "sports",
    "baseball bat": "sports",
    "baseball glove": "sports",
    "skateboard": "sports",
    "surfboard": "sports",
    "tennis racket": "sports",
    
    # Activities/Cooking
    "bottle": "cooking",
    "wine glass": "cooking",
    "cup": "cooking",
    "fork": "cooking",
    "knife": "cooking",
    "spoon": "cooking",
    "bowl": "cooking"
}

def generate_mapping_report(output_dir: str = "output"):
    """Generates a summary of which COCO classes map to which FGT classes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # FGT classes covered
    covered_fgt = set(COCO_TO_FGT.values())
    
    report = {
        "mapping": COCO_TO_FGT,
        "fgt_classes_covered": list(covered_fgt),
        "fgt_classes_covered_count": len(covered_fgt),
        "unmapped_or_weakly_mapped_fgt_classes": [
            "selfie", "group_photo", "crowd",
            "beach", "mountain", "city", "rural",
            "celebration", "work", "travel",
            "art",
            "screenshot", "receipt", "id_card", "handwritten", "printed",
            "landscape", "sky", "water",
            "wedding", "birthday", "concert", "graduation", "holiday"
        ]
    }
    
    report_path = os.path.join(output_dir, "bootstrap_mapping_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Mapping report saved to {report_path}")
    print(f"Mapped {len(COCO_TO_FGT)} COCO classes to {len(covered_fgt)} FGT taxonomy leaves.")
    print("Notice: Many gallery-specific concepts (e.g. selfies, documents, events) lack COCO equivalents and remain unmapped.")

if __name__ == "__main__":
    generate_mapping_report()
