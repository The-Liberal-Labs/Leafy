# prepare_data.py
import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Path to your master folder containing the 116 disease folders
SOURCE_DATA_DIR = "./data" 
# Path where you want to create the train/val/test split
OUTPUT_DIR = "./new_data" 
# Train, validation, and test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO will be the remainder (0.10)

# --- Main Logic ---
if os.path.exists(OUTPUT_DIR):
    print(f"Output directory {OUTPUT_DIR} already exists. Deleting it.")
    shutil.rmtree(OUTPUT_DIR)

print("Creating new output directory structure...")
train_path = os.path.join(OUTPUT_DIR, "train")
val_path = os.path.join(OUTPUT_DIR, "val")
test_path = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

disease_classes = [d for d in os.listdir(SOURCE_DATA_DIR) if not d.startswith('.')]
print(f"Found {len(disease_classes)} classes.")

for disease_class in disease_classes:
    print(f"Processing class: {disease_class}")
    
    # Create class subdirectories in train, val, and test
    os.makedirs(os.path.join(train_path, disease_class), exist_ok=True)
    os.makedirs(os.path.join(val_path, disease_class), exist_ok=True)
    os.makedirs(os.path.join(test_path, disease_class), exist_ok=True)
    
    # Get all image files for the current class
    class_dir = os.path.join(SOURCE_DATA_DIR, disease_class)
    if not os.path.isdir(class_dir):
        continue
        
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    
    # ✨ THE FIX: Add a guard clause here ✨
    # Check if the list of images is empty before trying to split it.
    if not images:
        print(f"  [WARNING] No images found for class '{disease_class}'. Skipping.")
        continue # Move to the next class

    # If the script reaches here, we know we have images to split.
    # The rest of the code can now run safely.
    
    # First, split into training and (validation + test)
    train_images, val_test_images = train_test_split(images, test_size=(1.0 - TRAIN_RATIO), random_state=42)
    
    # Now split (validation + test) into validation and test
    # The new test_size is relative to the val_test_images set
    relative_test_size = VAL_RATIO / (1.0 - TRAIN_RATIO)

    # ✨ ANOTHER GUARD CLAUSE: Handle cases with very few images ✨
    # If there's only one image in the val_test set, we can't split it.
    if len(val_test_images) < 2:
        print(f"  [WARNING] Not enough images for class '{disease_class}' to create a validation/test split. Placing all in training.")
        # In this case, we'll just copy all images to the training set.
        train_images = images
        val_images = []
        test_images = []
    else:
        val_images, test_images = train_test_split(val_test_images, test_size=relative_test_size, random_state=42)
    
    # Function to copy files
    def copy_files(file_list, destination_folder):
        for file_name in file_list:
            source_file = os.path.join(class_dir, file_name)
            dest_file = os.path.join(destination_folder, disease_class, file_name)
            shutil.copy2(source_file, dest_file)

    # Copy files to their new homes
    copy_files(train_images, train_path)
    copy_files(val_images, val_path)
    copy_files(test_images, test_path)

print("\nData splitting complete!")
print(f"Data has been split into train, val, and test sets in: {OUTPUT_DIR}")