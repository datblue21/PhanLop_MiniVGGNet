from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Load and preprocess the dataset
def load_dataset(dataset_path):
    data = []
    labels = []

    # Grab the image1 paths and randomly shuffle them
    image_paths = list(paths.list_images(dataset_path))
    np.random.shuffle(image_paths)

    for image_path in image_paths:
        # Load the image1 and extract the class label
        image = cv2.imread(image_path)
        label = image_path.split(os.path.sep)[-2]

        # Preprocess the image1
        image = cv2.resize(image, (32, 32))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)

    # Convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels

# Define the MiniVGGNet model
def build_mini_vggnet(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    chan_dim = -1

    # First CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # First (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Load the dataset
print("[INFO] loading images...")
data, labels = load_dataset(args["dataset"])

# Convert the labels from integers to vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels, 5)  # Change 5 to the number of flower classes

# Partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Construct the image1 generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = build_mini_vggnet(width=32, height=32, depth=3, classes=5)  # Change 5 to the number of flower classes
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 32,
    epochs=40,
    verbose=1
)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
