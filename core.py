from keras import layers
from keras.models import Sequential

from train_test import X_train, X_test, y_train, y_test


model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(26,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_train, epochs=1500, batch_size=128)

model.evaluate(x=X_test, y=y_test, batch_size=128)

model.save('model.h5')
