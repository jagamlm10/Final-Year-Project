import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras import layers , models


# Step 1: Load the CIFAKE dataset
def load_cifake_dataset(data_dir , subset):
    """
    Load the real and fake images from the train/test directories.
    :param data_dir: Path to the cifake-real-and-ai-generated-synthetic-images folder.
    :param subset: 'train' or 'test' to specify which subset of data to load.
    :return: X (images), y (labels)
    """
    subset_dir = os.path.join(data_dir , subset)  # train or test directory
    real_dir = os.path.join(subset_dir , 'REAL')
    fake_dir = os.path.join(subset_dir , 'FAKE')

    real_images = []
    fake_images = []

    # Load real images (searching for .jpg files)
    for img_path in glob.glob(real_dir + '/*.jpg'):
        img = Image.open(img_path).convert('RGB').resize((32 , 32))  # Ensure RGB format
        img = np.array(img) / 255.0  # Normalize
        real_images.append(img)

    # Load fake images (searching for .jpg files)
    for img_path in glob.glob(fake_dir + '/*.jpg'):
        img = Image.open(img_path).convert('RGB').resize((32 , 32))
        img = np.array(img) / 255.0
        fake_images.append(img)

    # Convert to numpy arrays
    real_images = np.array(real_images)
    fake_images = np.array(fake_images)

    # Print shape of loaded images
    print(f"Shape of real_images: {real_images.shape}")
    print(f"Shape of fake_images: {fake_images.shape}")

    # Ensure that images are reshaped to (32, 32, 3)
    if len(real_images.shape) != 4 or real_images.shape[1:] != (32 , 32 , 3):
        real_images = real_images.reshape((-1 , 32 , 32 , 3))
    if len(fake_images.shape) != 4 or fake_images.shape[1:] != (32 , 32 , 3):
        fake_images = fake_images.reshape((-1 , 32 , 32 , 3))

    # Print shape after reshaping
    print(f"After reshaping: real_images shape = {real_images.shape}, fake_images shape = {fake_images.shape}")

    # Create labels: 0 for real, 1 for fake
    real_labels = np.zeros((real_images.shape[0] , 1))
    fake_labels = np.ones((fake_images.shape[0] , 1))

    # Combine the datasets
    X = np.concatenate([real_images , fake_images] , axis=0)
    y = np.concatenate([real_labels , fake_labels] , axis=0)

    return X , y


# Step 2: Load the CIFAKE dataset from both 'train' and 'test' sets
data_dir = './cifake-real-and-ai-generated-synthetic-images'

# Load training data
X_train , y_train = load_cifake_dataset(data_dir , 'train')

# Load test data
X_test , y_test = load_cifake_dataset(data_dir , 'test')

# Ensure that X_train and X_test have the correct shape (32, 32, 3)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
assert X_train.shape[1:] == (32 , 32 , 3) , "Train data has incorrect shape"
assert X_test.shape[1:] == (32 , 32 , 3) , "Test data has incorrect shape"

# Step 3: Build the CNN model
model = models.Sequential()

# First convolutional block
model.add(layers.Conv2D(32 , (3 , 3) , activation='relu' , input_shape=(32 , 32 , 3)))
model.add(layers.MaxPooling2D((2 , 2)))

# Second convolutional block
model.add(layers.Conv2D(64 , (3 , 3) , activation='relu'))
model.add(layers.MaxPooling2D((2 , 2)))

# Third convolutional block
model.add(layers.Conv2D(64 , (3 , 3) , activation='relu'))

# Flatten the output and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64 , activation='relu'))
model.add(layers.Dense(1 , activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam' ,
              loss='binary_crossentropy' ,
              metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train , y_train , epochs=10 , validation_data=(X_test , y_test))

# Step 5: Evaluate the model
test_loss , test_acc = model.evaluate(X_test , y_test , verbose=2)
print(f'Test accuracy: {test_acc * 100} %')
