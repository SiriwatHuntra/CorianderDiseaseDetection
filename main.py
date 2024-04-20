from plant_detector import PlantDetector
from plant_classifier import PlantClassifier
from ModelMaker import PlantTrainer
from folderCleaner import clear_folders

input_directory = "test_img"
buffer_directory = "buffer_folder"
input_folder = "buffer_folder"
output_folders = ["classed/h", "classed/u", "classed/n", "classed/w"]
data_dirs = ["trainData/h", "trainData/u", "trainData/n", "trainData/w"]

confidence = 0.8

# Create an instance of PlantDetector
detector = PlantDetector()
classifier = PlantClassifier()
trainer = PlantTrainer()

# Call the function
detector.image_crops(input_directory, buffer_directory, confidence)

classifier.classify_images(input_folder, output_folders)

#trainer.train_model(data_dirs)

#clear_folders(output_folders)
