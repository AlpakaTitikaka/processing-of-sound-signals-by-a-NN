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
