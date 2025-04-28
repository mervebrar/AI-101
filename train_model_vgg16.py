import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# Veri artırma (train için)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

# Validation ve Test için sadece normalize
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Verileri yükle
train_generator = train_datagen.flow_from_directory(
    '/Users/merveebrardemirel/Desktop/practice_datasets/processed_catdogdata/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    '/Users/merveebrardemirel/Desktop/practice_datasets/processed_catdogdata/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    '/Users/merveebrardemirel/Desktop/practice_datasets/processed_catdogdata/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Test seti için shuffle kapalı olsun (doğru eşleşme için)
)

# VGG16 Modeli yükle (ImageNet ağırlıklarıyla)
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Base modeli donduralım
for layer in base_model.layers:
    layer.trainable = False

# Üstüne kendi katmanlarımızı ekleyelim
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Modeli tanımla
model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator  # Buraya dikkat! Test değil, validation verisi kullanıyoruz
)

# Eğitim grafiği
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('VGG16 Model Training Progress')
plt.show()

# Modeli kaydet
model.save('cat_dog_classifier_vgg16.h5')

# Test verisinde tahmin yap (şimdi test setiyle ölçüm yapıyoruz)
test_generator.reset()
pred = model.predict(test_generator, verbose=1)
predicted_classes = (pred > 0.5).astype('int32').flatten()

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification Report
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
