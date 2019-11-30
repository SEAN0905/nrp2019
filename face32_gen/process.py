# the script is to preprocess the dataset to convert it to 64*64 
# grayscale image with short naming
# 
from matplotlib import pyplot as plt
from PIL import Image as Im
import os

base_path = ""

# # # classify images to different folders based on smile.txt label
# # read smile labels
# broken_series = 0
# handle = open("smile.txt", "r")
# smile_label = []
# for line in handle:
#     data = line.strip()
#     if data == "0" or data == "1":
#         smile_label.append(data)
# handle.close()

# # manually classify
# for i in range(1, 3986+1):
#     file = base_path + "image_smile/" + str(i) + ".jpg"
#     if smile_label[i-1] == "0":
#         new_file = base_path + "image_smile/0/" + str(i) + ".jpg"
#     elif smile_label[i-1] == "1":
#         new_file = base_path + "image_smile/1/" + str(i) + ".jpg"
#     try:
#         Im.open(file).save(new_file)
#     except:
#         broken_series += 1
#         continue

# print(broken_series)

# this is to move those wrongly classified to the correct folder
for i in range(1600, 2154+1):
    file = base_path + "image_smile/1/" + str(i) + ".jpg"
    new_file = base_path + "image_smile/0/" + str(i) + ".jpg"
    try:
        Im.open(file).save(new_file)
    except:
        print("Awwww man")

