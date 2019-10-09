from keras.models import load_model
import cv2
import numpy as np

def predict_from_spectogram(model_path='model_150_50.h5', img_path='D:/daizwoz_depression_dataset/data/img/test/depression/355_2.png'):
    model = load_model(model_path)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])

    classes = model.predict_classes(img)
    if classes[0][0] == 1:
        print('normal')
        return 'normal'
    else:
        print('depressed')
        return 'depressed'



model = load_model('model_150_50.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('D:/daizwoz_depression_dataset/data/img/test/depression/355_2.png')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict_classes(img)
if classes[0][0] == 1:
    print('normal')
else:
    print('depressed')
# print(classes[0][0])