from PIL import Image
import numpy as np

file = "face32_relabeled/image/1.jpg"
image = Image.open(file)
image.load()
raw_image = np.asarray(image, dtype="int32")
print(raw_image)

img = np.reshape(raw_image, (32, 32))
new_image = Image.fromarray(np.uint8(img), "L")
new_image.save("test_image.jpg")
new_image.show()