# Bu kodda klasordeki resimleri sentetik olarak bozup FVC2002DB1_Synthetic_Distorted dosyasina yaziyoruz.

import os
import cv2
from noise_generator import noise_generator

input_file_path = "FVC2002DB1"
output_file_path = "FVC2002DB1_Synthetic_Distorted"
noise_types = ["gauss", "saltNpepper", "concave"]

try:
    # Create target Directory
    os.mkdir(output_file_path)
    print("Directory ", output_file_path, " Created ")
except FileExistsError:
    print("Directory ", output_file_path, " already exists")

try:
    # Create target Directory
    for noises in noise_types:
        os.mkdir(os.path.join(output_file_path, noises))
        print("Directory ", output_file_path + noises, " Created ")
except FileExistsError:
    print("Directory for synthhetic distortions are already exists")

# List of folder names, indicate number of classes
user_list = os.listdir(input_file_path)
user_count = len(user_list)

for user_dir in os.listdir(input_file_path):
    for noise in noise_types:
        print(noise)
        try:
            # Create target Directory
            os.mkdir(os.path.join(output_file_path, noise, user_dir))
            print("Directory ", os.path.join(output_file_path, noise, user_dir), " Created ")
        except FileExistsError:
            print("Directory ", os.path.join(output_file_path, noise, user_dir), " already exists")

        img_list = os.listdir(os.path.join(input_file_path, user_dir))
        for img_dir in img_list:
            image = cv2.imread(os.path.join(input_file_path, user_dir, img_dir), cv2.IMREAD_GRAYSCALE)
            noisy_image = noise_generator(noise, image)
            cv2.imwrite(os.path.join(output_file_path, noise, user_dir, img_dir), noisy_image)



