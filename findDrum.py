import shutil
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import os

trains_dirname = os.path.join('dataset', 'labeled_drum', 'trains')
valids_dirname = os.path.join('dataset', 'labeled_drum', 'valids')

trains_generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=20.0,
    # shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

valids_generator = ImageDataGenerator()

trains_generator = trains_generator.flow_from_directory(
    trains_dirname,
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary'
)

valids_generator = valids_generator.flow_from_directory(
    valids_dirname,
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary'
)


model = Sequential()

model.add(SeparableConv2D(32, kernel_size=3, padding='same',
                          input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SeparableConv2D(32, kernel_size=3,
                          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SeparableConv2D(16, kernel_size=3,
                          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SeparableConv2D(8, kernel_size=5, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['acc', 'mae'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'mae'])

model.summary()

model_filename = os.path.join(
    'models', 'finding_drum_model_{val_loss:.2f}.h5')

model.fit_generator(
    trains_generator,
    validation_data=valids_generator,
    steps_per_epoch=100,
    validation_steps=100,
    epochs=100,
    callbacks=[
        TensorBoard(log_dir='tflogs'),
        EarlyStopping(patience=3, monitor='val_loss'),
        ModelCheckpoint(model_filename, monitor='val_loss',
                        save_best_only=True),
    ]
)

predicting_dirname = os.path.join('dataset', 'reshaped_drum', '*')
predicted_dirname = os.path.join('dataset', 'predicted_drum')
for i, file in enumerate(glob.glob(predicting_dirname)):
    image = load_img(file).resize((32, 32))
    array = img_to_array(image) / 255
    predicted = model.predict(np.array([
        array
    ]))
    prob = int(predicted[0][0]*100)
    print(f'file={file}, drum?={prob}%')

    if prob >= 95:
        shutil.copy(file, predicted_dirname)
