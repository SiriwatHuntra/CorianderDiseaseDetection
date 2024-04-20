import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from collections import Counter
from datetime import datetime

class PlantClassifier:
    def __init__(self, model_path="plant_health_classifier.keras"):
        self.model = load_model(model_path)

    def classify_images(self, input_folder, output_folders):
        # Ensure output directories exist
        for folder in output_folders:
            os.makedirs(folder, exist_ok=True)

        # Initialize a Counter object to keep track of class counts
        class_counts = Counter()

        # Define a dictionary to map folder names to class names
        class_names = {
            'classed/h': 'Healthy Plant',
            'classed/u': 'Unhealthy Plant',
            'classed/n': 'Not a Target Plant',
            'classed/w': 'Weed in Pot'
        }

        # Process each image in the input folder
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize to match model's expected input shape
            img = img / 255.0  # Normalize pixel values to [0, 1]
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Use the model to predict the class
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)  # Convert one-hot encoded vector to class index

            # Increment the count for the predicted class
            class_counts[class_names[output_folders[predicted_class[0]]]] += 1

            # Move the image to the corresponding output folder
            output_folder = output_folders[predicted_class[0]]
            output_path = os.path.join(output_folder, img_name)
            os.rename(img_path, output_path)

        print("Image classification completed! Images have been moved to their respective class folders.")

        # Create a DataFrame from the class counts
        df = pd.DataFrame.from_records([class_counts], columns=class_names.values())

        # Add the current date and time
        df.insert(0, 'date_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Check if the CSV file exists
        if os.path.isfile('class_counts.csv'):
            # If it exists, append without the header
            df.to_csv('class_counts.csv', mode='a', header=False, index=False)
        else:
            # If it doesn't exist, create a new file with a header
            df.to_csv('class_counts.csv', index=False)
