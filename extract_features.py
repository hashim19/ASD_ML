import numpy as np
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, meshgrid, ceil, linspace
import math
import pandas as pd
import soundfile as sf
import os
import pickle

from feature_functions import extract_cqcc, extract_lfcc


def extract_features(file, features, data_type='train', data_label='bonafide', feat_root='Features', cached=False):

    def get_feats():
        data, samplerate = sf.read(file)

        if features == 'cqcc':
            return extract_cqcc(data, samplerate)

        if features == 'lfcc':
            return extract_lfcc(data, samplerate)

        else:
            return None

    if cached:

        if data_type == 'train':

            feat_dir = os.path.join(feat_root, features + '_features', data_type, data_label)
        
        else:

            feat_dir = os.path.join(feat_root, features + '_features', data_label)

        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        feat_file = file.split('.')[0].split('/')[-1] + '.pkl'

        feat_fullfile = feat_dir + '/' + feat_file

        # print(feat_fullfile)

        if not os.path.exists(feat_fullfile):

            feat_data = get_feats()

            with open(feat_fullfile, 'wb') as f:
                pickle.dump(feat_data, f)

        else:

            with open(feat_fullfile, 'rb') as f:
                feat_data = pickle.load(f)
        
        return feat_data

    return get_feats()


if __name__ == "__main__":

    db_folder = '/home/hashim/PhD/Data/AsvSpoofData_2019/train/'
    data_dirs = [db_folder + 'LA/ASVspoof2019_LA_train/flac/']  # [db_folder + 'LA/ASVspoof2019_LA_train/flac/', db_folder + 'LA/ASVspoof2019_LA_dev/flac/']
    protocol_paths = [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt']  # [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt']

    Feat_dir = 'features_out'

    audio_ext = '.flac'

    data_type = 'train'

    data_labels = ['bonafide', 'spoof']

    features = 'cqcc'



    # extract features and save them
    for k, protocol_path in enumerate(protocol_paths):

        for data_label in data_labels:

            df = pd.read_csv(protocol_path, sep=' ', header=None)
            files = df[df[4] == data_label][1]
            print("{} data size is {}".format(data_label, files.shape))

            for nf, file in enumerate(files):
                Tx = extract_features(data_dirs[k] + file + audio_ext, features=features, data_label=data_label,
                 data_type=data_type, feat_root=Feat_dir, cached=True)
                print(Tx.shape)