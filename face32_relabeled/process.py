# the script is to preprocess the dataset to convert it to 64*64 
# grayscale image with short naming
# 
from matplotlib import pyplot as plt
from PIL import Image as Im
import os

base_path = "image_gender/"

# # # # classify images to different folders based on smile.txt label
# # # read smile labels
# # broken_series = 0
# # handle = open("smile.txt", "r")
# # smile_label = []
# # for line in handle:
# #     data = line.strip()
# #     if data == "0" or data == "1":
# #         smile_label.append(data)
# # handle.close()

# # # manually classify
# # for i in range(1, 3986+1):
# #     file = base_path + "image_smile/" + str(i) + ".jpg"
# #     if smile_label[i-1] == "0":
# #         new_file = base_path + "image_smile/0/" + str(i) + ".jpg"
# #     elif smile_label[i-1] == "1":
# #         new_file = base_path + "image_smile/1/" + str(i) + ".jpg"
# #     try:
# #         Im.open(file).save(new_file)
# #     except:
# #         broken_series += 1
# #         continue

# # print(broken_series)

# # this is to move those wrongly classified to the correct folder
# for i in range(1600, 2154+1):
#     file = base_path + "image_smile/1/" + str(i) + ".jpg"
#     new_file = base_path + "image_smile/0/" + str(i) + ".jpg"
#     try:
#         Im.open(file).save(new_file)
#     except:
#         print("Awwww man")

# this block is to record labels for gender
gender_label = []
counter = 0
for i in range(1, 2723+1):
    path1 = base_path + "0/" + str(i) + ".jpg"
    path2 = base_path + "1/" + str(i) + ".jpg"
    # # naming is from 1 to 2723
    # if os.path.exists(path1) and os.path.exists(path2):
    #     print("error", i)
    if os.path.exists(path1):
        counter += 1
        gender_label.append(0)
        # proper_name = base_path + "0/" + str(counter) + ".jpg"
        # os.rename(path1, proper_name)
    elif os.path.exists(path2):
        counter += 1
        gender_label.append(1)
        # proper_name = base_path + "1/" + str(counter) + ".jpg"
        # os.rename(path2, proper_name)
    else:
        print("error2", i)

# handle_gender = open("gender.txt", "w")
# for i in range(2723):
#     # print(gender_label[i])
#     handle_gender.write(str(gender_label[i])+"\n")
# handle_gender.close()

# # this block is to record labels for smile
# smile_label = []
# counter = 0
# for i in range(1, 2723+1):
#     path1 = base_path + "0/" + str(i) + ".jpg"
#     path2 = base_path + "1/" + str(i) + ".jpg"
    
#     if os.path.exists(path1):
#         counter += 1
#         # smile_label.append(0)
#         proper_name = base_path + "0/" + str(counter) + ".jpg"
#         os.rename(path1, proper_name)
#     elif os.path.exists(path2):
#         counter += 1
#         # smile_label.append(1)
#         proper_name = base_path + "1/" + str(counter) + ".jpg"
#         os.rename(path2, proper_name)
#     else:
#         print("error2", i)
handle_smile = open("smile.txt", "w")
for _ in range(1, 1532+1):
    handle_smile.write("0\n")
for _ in range(1533, 2723+1):
    handle_smile.write("1\n")
handle_smile.close()
