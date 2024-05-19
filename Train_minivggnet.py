# import các thư viện
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import BatchNormalization, Activation
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

# Bước 1. Chuẩn bị dữ liệu
# Khởi tạo tiền xử lý ảnh
sp = SimplePreprocessor(32, 32)  # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor()  # Gọi hàm để chuyển ảnh sang mảng

print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets"))  # # tạo danh sách đường dẫn đến các folder con của folder datasets

# Nạp ảnh rồi chuyển mức xám của pixel trong vùng [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)  # Mỗi lần nạp 500 file
data = data.astype("float") / 255.0

# Chia tách dữ liệu vào 02 tập, training: 75% và testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Khởi tạo danh sách các label cho tập dữ liệu animals
label_names = ["cat", "dog", "panda"]

# Data augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply data augmentation to training data
trainX = data_generator.flow(trainX, trainY, batch_size=32)

# Bước 2. Xây dựng cấu trúc model (mạng)

print("[INFO]: Biên dịch model....")
# Các tham số bộ tối ưu:
#   - learning_rate: Tốc dộ học
#   - decay: sử dụng để giảm từ từ tốc độ học theo thời gian
#            được tính bằng Tốc độ học /tổng epoch. Dùng để
#            tránh overfitting và tăng độ chính xác khi tranning
#   - momentum: Hệ số quán tính-opl;,.
#   - nesterov = True: sử dụng phương pháp tối ưu Nestrov accelerated gradient

# Tạo bộ tối ưu hóa cho model (hàm tối ưu SGD)
optimizer = SGD(learning_rate=0.001, decay=0, momentum=0.9, nesterov=True)
# optimizer = Adam(learning_rate=0.001)


# Tạo model (mạng), biên dịch model
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


# Learning rate scheduler
# def lr_schedule(epoch==60):
#     return 0.01 * (0.1 ** (epoch // 10))
#
# lr_scheduler = LearningRateScheduler(lr_schedule)
#
# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler function adjusted for 60 epochs
def lr_schedule(epoch=60):
    # Reduces the learning rate after 1/3rd and 2/3rd of total epochs
    if epoch < 20:
        return 0.001  # Initial learning rate
    elif epoch < 40:
        return 0.0001  # Reduce to 0.001 after 20 epochs
    else:
        return 0.00001  # Reduce to 0.0001 after 40 epochs


learning_rate_scheduler = LearningRateScheduler(lr_schedule)
# Early stopping adjusted for 60 epochs training
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Bước 3: Thực hiện Train model/network
print("[INFO]: Đang trainning....")
# H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=60, verbose=1)
H = model.fit(trainX, validation_data=(testX, testY), batch_size=32, epochs=60, verbose=1)

# lưu model
# model.save("miniVGGNet.hdf5")  # Lưu model
# model.summary()  # Hiển thị tóm tắt các tham số của model

# Bước 4: Đánh giá mạng
print("[INFO]: Đánh giá model....")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# Vẽ biểu đồ
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 60), H.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 60), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 60), H.history["accuracy"], label="Độ chính xác khi trainning")
plt.plot(np.arange(0, 60), H.history["val_accuracy"], label="val_acc")
plt.title("Giá trị Loss và độ chính xác trên tập flower")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()
