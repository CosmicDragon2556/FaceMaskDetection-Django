from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class ModelTrainer:
    @staticmethod
    def get_class_weights(class_counts):
        total = sum(class_counts.values())
        return {i: total/(len(class_counts)*count) 
                for i, (cls, count) in enumerate(class_counts.items())}

    @staticmethod
    def train_model(model, dataset_path, class_weights):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        history = model.fit(
            train_generator,
            epochs=20,
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(patience=5),
                ModelCheckpoint('best_model.h5', save_best_only=True)
            ]
        )
        return history