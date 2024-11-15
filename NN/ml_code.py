import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

path = '/Users/user/Documents/Project/___e-zest___/Dashboard_division/NN/data/myData'
test_ratio = 0.2
validation_ratio = 0.2
image_dimensions = (32, 32, 3)

# Load images
myList = os.listdir(path)
if '.DS_Store' in myList:
    myList.remove('.DS_Store')
print("Folders:", myList)
noofdigits = len(myList)

images = []
digitno = []

for folder in range(noofdigits):
    myPicList = os.listdir(os.path.join(path, str(folder)))
    for image in myPicList:
        curImg = cv2.imread(os.path.join(path, str(folder), image))
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        digitno.append(folder)
    print(f"Processed folder: {folder}")

images = np.array(images)
classNo = np.array(digitno)

# Split data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

# Preprocess images
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

x_train = np.array(list(map(preprocessing, x_train)))
x_test = np.array(list(map(preprocessing, x_test)))
x_validation = np.array(list(map(preprocessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

# One-hot encoding
y_train = to_categorical(y_train, noofdigits)
y_test = to_categorical(y_test, noofdigits)
y_validation = to_categorical(y_validation, noofdigits)

# Data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             rotation_range=10,
                             shear_range=0.1)
dataGen.fit(x_train)

# Model creation
def myModel():
    noofFilters = 60
    sizeofFilter1 = (5, 5)
    sizeofFilter2 = (3, 3)
    sizeofPool = (2, 2)
    noofNode = 500
    model = Sequential()
    model.add(Conv2D(noofFilters, sizeofFilter1, input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(noofFilters, sizeofFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Conv2D(noofFilters // 2, sizeofFilter2, activation='relu'))
    model.add(Conv2D(noofFilters // 2, sizeofFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noofNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noofdigits, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 10
stepsperEpoch = len(x_train) // batchSizeVal

model.fit(dataGen.flow(x_train, y_train, batch_size=batchSizeVal),
          steps_per_epoch=stepsperEpoch,
          epochs=epochsVal,
          validation_data=(x_validation, y_validation),
          shuffle=True)

# Save model
model.save('model/graph_recognisor2.h5')
