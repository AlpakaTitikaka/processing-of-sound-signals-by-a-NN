#This file is part of processing-of-sound-signals-by-a-NN.

#processing-of-sound-signals-by-a-NN is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#processing-of-sound-signals-by-a-NN is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.


import librosa
import csv
import numpy as np
import os


def sound_converter(songname):
    y, sr = librosa.load(songname, mono=True, duration=30)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc


def create_data_file(name, header):
    file = open(name, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    return 0


def update_data_file(name, to_append):
    file = open(name, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


def readdata():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    filepath = os.path.abspath('sounds')
    name = 'input.csv'

    create_data_file(name, header)

    for filename in os.listdir(filepath):

        if filename.endswith('.wav'):

            songname = str(f'{filepath}/{filename}')

            chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc = sound_converter(songname)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
            for e in mfcc:
                to_append += f' {np.mean(e)}'

            update_data_file(name, to_append)

