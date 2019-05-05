"""
Bu program 'FVC2002DB1' klasöründeki resimleri alıp 'FVC2002DB1_Generated' adlı yeni bir klasöre
oluşturulan yeni resimleri kaydeder.
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

NUM_OF_COPY_IMAGE = 100

datagen = ImageDataGenerator(
        rotation_range=120,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        shear_range=0.0,
        fill_mode='constant',
        cval=255)

path = os.path.join("FVC2002DB1")
pathGenerated = os.path.join("FVC2002DB1_Generated")

try:
    # Create target Directory
    os.mkdir(pathGenerated)
    print("Directory ", pathGenerated,  " Created ")
except FileExistsError:
    print("Directory ", pathGenerated,  " already exists")

for personID in os.listdir(path):
    try:
        # Create target Directory
        os.mkdir(os.path.join(pathGenerated, personID))
        print("Directory ", os.path.join(pathGenerated, personID),  " Created ")
    except FileExistsError:
        print("Directory ", os.path.join(pathGenerated, personID),  " already exists")

    for fingerID in os.listdir(os.path.join(path, personID)):
        img = load_img(os.path.join(path, personID, fingerID))  # PIL image
        x = img_to_array(img)  # Numpy array with shape (1, 388, 374)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 1, 388, 374)

        i = 0
        prefix = fingerID.split('.')
        for batch in datagen.flow(x, batch_size=1, shuffle=False, save_to_dir=os.path.join(pathGenerated, personID), save_prefix= prefix[0], save_format='jpeg'):
            i += 1
            if i > NUM_OF_COPY_IMAGE-1:  # Generating NUM_OF_COPY_IMAGE images from 1 sample.
                break
print("Image Data Generation is done.")
