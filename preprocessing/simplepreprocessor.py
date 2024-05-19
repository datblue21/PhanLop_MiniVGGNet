# # import the necessary packages
# import cv2
# import numpy as np
#
# class SimplePreprocessor:
#     def __init__(self, width, height, inter=cv2.INTER_AREA):
#         # Lưu image width, height và interpolation
#         self.width = width
#         self.height = height
#         self.inter = inter
#
#         # Tính mean của ImageNet dataset (để subtraction)
#         self.mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#
#     def preprocess(self, image):
#         # Trả về ảnh có kích thước đã thay đổi
#         image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
#
#         # Mean subtraction
#         image = image.astype("float32")
#         image = image - self.mean
#
#         # Normalization
#         image = np.clip(image, 0, 255)  # Clip pixel values to [0, 255]
#         image /= 255.0  # Normalize pixel values to [0, 1]
#
#         return image


# import the necessary packages
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # Lưu image1 width, height và interpolation
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # Trả về ảnh có kích thước đã thay đổi
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
