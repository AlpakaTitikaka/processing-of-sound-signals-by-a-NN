#This file is part of processing-of-sound-signals-by-a-NN.

#processing-of-sound-signals-by-a-NN is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#processing-of-sound-signals-by-a-NN is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.


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
