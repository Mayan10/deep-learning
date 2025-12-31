import os
import shutil
import random

DATASET_DIR = "PetImages"
CATEGORIES = ["Cat", "Dog"]
TEST_DIR = os.path.join(DATASET_DIR, "Testing")
TEST_SPLIT = 0.2  # 20% test data

random.seed(42)

# Create Testing folders
for category in CATEGORIES:
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

for category in CATEGORIES:
    source_dir = os.path.join(DATASET_DIR, category)
    test_category_dir = os.path.join(TEST_DIR, category)

    images = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]

    test_size = int(len(images) * TEST_SPLIT)
    test_images = random.sample(images, test_size)

    for img in test_images:
        src = os.path.join(source_dir, img)
        dst = os.path.join(test_category_dir, img)
        shutil.copy2(src, dst)

print("Test dataset created successfully.")
