# the script is to preprocess the dataset to convert it to 64*64 
# grayscale image with short naming
# 
from matplotlib import pyplot as plt
from PIL import Image as Im
import os

base_path = "/Users/ue/Downloads/nrp2019/face32/"

# function to convert 1 to "file0000000000000001"
def convertN2S(integer):
    result = "file"
    for i in range(16-len(str(integer))):
        result += "0"
    result += str(integer)
    return result

# # rename the images, from "file0000000000000001.jpg" to "1.jpg"
# count = 1
# broken_series = 0
# for i in range(1, 3987):
#     file_old_name = base_path + "image/" + convertN2S(i) + ".jpg"
#     file_new_name = base_path + "image/" + str(i) + ".jpg"
#     try:
#         os.rename(file_old_name, file_new_name)
#         # os.rename(file_new_name, file_old_name)
#         count += 1
#     except:
#         broken_series += 1
#         continue
#     # print(file_old_name)
#     # print("file0000000000000001.jpg")
# print(count)
# print(broken_series)

# now downscale the image to 64*64 grayscale image
broken_series = 0
for i in range(1, 3986+1):
    file = base_path + "image/0/" + str(i) + ".jpg"
    try:
        Im.open(file).convert('L').resize((32, 32), Im.ANTIALIAS).save(file)
    except:
        broken_series += 1
        continue

for i in range(1, 3986+1):
    file = base_path + "image/1/" + str(i) + ".jpg"
    try:
        Im.open(file).convert('L').resize((32, 32), Im.ANTIALIAS).save(file)
    except:
        broken_series += 1
        continue

print(broken_series)