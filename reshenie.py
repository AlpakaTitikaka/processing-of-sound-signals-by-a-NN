#This file is part of MoiTrudi.

#MoiTrudi is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#MoiTrudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd
import os

from util import readdata
from train_test import encoder, scaler

from keras.models import load_model


readdata()

data = pd.read_csv('input.csv')
data.head()
k = len(data)

os.remove('input.csv')

genre_list = data.iloc[:, 0]
data = data.drop(['filename'], axis=1)

input_set = scaler.fit_transform(np.array(data, dtype=float))

model = load_model('model.h5')
pred = model.predict(input_set, batch_size=128)

predic = np.argmax(pred, axis=1)

goals = genre_list
prediction = encoder.inverse_transform(predic)

for i in range(k):
    print(goals[i], ' ', prediction[i], '\n')
