# plant_detector.py
from PIL import Image, ImageOps
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import os
import datetime

class PlantDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", target_labels=["potted plant"]):
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.target_labels = target_labels
        self.counts = {}

    def image_crops(self, input_directory, output_directory, confidence):
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        for filename in os.listdir(input_directory):
            curr_path = os.path.join(input_directory, filename)
            image = Image.open(curr_path)
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]
            with Image.open(curr_path) as im:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    label_text = self.model.config.id2label[label.item()]
                    if label_text in self.target_labels:
                        print(f"Detected {self.model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
                        self.counts[label_text] = self.counts.get(label_text, 0) + 1
                        remote_region = im.crop(box)
                        remote_region_resized = ImageOps.fit(remote_region, (1024, 1024), Image.ANTIALIAS)
                        # Rename the output image with date and order
                        date_str = datetime.datetime.now().strftime("%d%m%y_%H%M")
                        count_str = str(self.counts[label_text]).zfill(3)  # Pad with zeros
                        new_filename = f"{date_str}_{count_str}.jpg"
                        output_path = os.path.join(output_directory, new_filename)
                        remote_region_resized.save(output_path)