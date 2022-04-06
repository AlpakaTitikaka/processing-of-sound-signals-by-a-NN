#This file is part of processing-of-sound-signals-by-a-NN.

#processing-of-sound-signals-by-a-NN is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#processing-of-sound-signals-by-a-NN is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import os

import util


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

filepath = os.path.abspath('sounds')

util.create_data_file('dataset.csv', header)

genres = 'hunting fire tree_chop benzo'.split()

for g in genres:

    for filename in os.listdir(f'{filepath}\{g}'):

        if filename.endswith('.wav'):

            songname = str(f'{filepath}\{g}\{filename}')

            chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc = util.sound_converter(songname)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'

            util.update_data_file('dataset.csv', to_append)
