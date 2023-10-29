# Transfer learning is a machine learning technique where a model developed
# for a specific task is reused as the starting point for a model on a second task.
# Essentially, you're transferring the learned features (or representations) to a new problem.
# This is particularly useful in deep learning where training neural networks
# from scratch can be computationally expensive and time-consuming.

from keras.applications import VGG16  # this is a pre-trained model provided by keras
from keras import layers, Model
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from os import getcwd, sep

# Workflow Strategy:
# In a typical transfer learning workflow,
# you would first freeze all the pre-trained layers and add your custom layers on top.
# Then you'd train only these custom layers on your new dataset.
# Once the new layers have been trained to asatisfactory level,
# you can start to unfreeze some or all of the pre-trained layers and continue training,
# often with a much smaller learning rate. This two-phase approach allows the model to adapt to the new task
# while preserving the useful features learned during its original training.

# The result of not freezing the pretrained layers will be to destroy the information
# they contain during future training rounds. But if the new dataset is large enough,
# or similar to the used in pre-trained model,
# probably it's better to train the model with all the layers unfrozen.

def unfreeze_layers_and_fine_tuning(model):
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def freeze_layers(model):
    for layer in model.layers:
        layer.trainable = True

    return model


def build_model(freeze_pre_trained_layers=False):
    pre_trained = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    if freeze_pre_trained_layers:
        freeze_layers(pre_trained)

    # new layers
    x = layers.Flatten()(pre_trained.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation='softmax')(x)  # Assuming binary classification

    model = Model(pre_trained.input, x)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def fit_model(model, train_data, validation_data):
    model.fit(
        train_data,
        steps_per_epoch=train_data.samples // 32,
        validation_data=validation_data,
        validation_steps=validation_data.samples // 32,
        epochs=10
    )

    return model

def get_train_data():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    print(getcwd())
    train_generator = train_datagen.flow_from_directory(
        f'{getcwd()}{sep}train_images_transfer_learning{sep}train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        f'{getcwd()}{sep}train_images_transfer_learning{sep}validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, validation_generator


def train_model():
    train_data, validation_data = get_train_data()

    model_with_freezing_layers = build_model(freeze_pre_trained_layers=True)
    fit_model(model_with_freezing_layers, train_data, validation_data)
    unfreeze_layers_and_fine_tuning(model_with_freezing_layers)
    print('Model with freezing layers -> ', model_with_freezing_layers.metrics)
    print()

    model_without_freezing = build_model(freeze_pre_trained_layers=False)
    fit_model(model_without_freezing, train_data, validation_data)
    print('Model without freezing layers -> ', model_without_freezing.metrics)
    print()

if __name__ == '__main__':
    train_model()