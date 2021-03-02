from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False
# 훈련을 시킨다

vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))
