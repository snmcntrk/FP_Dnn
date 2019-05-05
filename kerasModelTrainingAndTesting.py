import os
import cv2
import keras
import numpy as np
from keras import models, layers, backend, utils
from keras_applications.densenet import DenseNet121


if __name__ == "__main__":
    input_dir = 'FVC2002DB1_Generated'
    #output_dir = 'newModelName.model'
    train_ratio = 0.80
    batch_size = 16
    input_size = (374, 388, 1)
    epochs = 20
    val_split = 0.25

    # List of folder names, indicate number of classes
    user_list = os.listdir(input_dir)
    user_count = len(user_list)

    print("User List:", user_list)
    print("User Count:", user_count)

    genericModel = DenseNet121(include_top=True,
                       input_shape=input_size,
                       classes=user_count,
                       weights=None,
                       backend=backend,
                       layers=layers,
                       utils=utils,
                       models=models)
    genericModel.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = []

    # Get paths of all images in given directory
    # Train and test split
    train_img_list = []
    test_img_list = []
    for user_dir in os.listdir(input_dir):
        img_list = os.listdir(os.path.join(input_dir, user_dir))
        train_count = int(len(img_list) * train_ratio)
        for i, img_dir in enumerate(img_list, 0):
            if i < train_count:
                train_img_list.append(os.path.join(input_dir, user_dir, img_dir))
            else:
                test_img_list.append(os.path.join(input_dir, user_dir, img_dir))
    # Shuffle them for training
    np.random.shuffle(train_img_list)
    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        for i in range(0, len(train_img_list), batch_size):
            X = []
            Y = []
            for user_dir in train_img_list[i:i + batch_size]:
                parent_dir, file_name = os.path.split(user_dir)
                _, user_folder_name = os.path.split(parent_dir)
                uix = user_list.index(user_folder_name)
                y = keras.utils.to_categorical(uix, num_classes=user_count, dtype='float32')
                Y.append(y)
                src_img = cv2.imread(user_dir, cv2.IMREAD_GRAYSCALE)
                X.append(src_img[:, :, np.newaxis])
            genericModel.fit(np.array(X), np.array(Y),
                       batch_size=batch_size,
                       epochs=1,
                       validation_split=val_split,
                       callbacks=callbacks)
    """
    print("Saving the model.")
    genericModel.save_weights(output_dir, overwrite=True)
    """
    print("Evaluating the model.")
    X = []
    Y = []
    print("Number of test data:", len(test_img_list))
    for img_dir in test_img_list:
        src_img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        parent_dir, file_name = os.path.split(img_dir)
        _, user_folder_name = os.path.split(parent_dir)
        X.append(src_img[:, :, np.newaxis])
        uix = user_list.index(user_folder_name)
        y = keras.utils.to_categorical(uix, num_classes=user_count, dtype='float32')
        Y.append(y)
    print("started...")
    scores = genericModel.evaluate(np.array(X), np.array(Y), batch_size=batch_size, verbose=True)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    print("Done !")
