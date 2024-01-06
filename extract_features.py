import numpy as np
import pandas as pd

from feature_functions import extract_cqcc, extract_lfcc


def extract_features(file, features, data_label='bonafide', cached=False):

    def get_feats():
        if features == 'cqcc':
            data, samplerate = sf.read(file)
            # cqcc_feat, fmax, fmin = extract_cqcc(data, samplerate)
            return extract_cqcc(data, samplerate)
        else:
            return None

    if cached:

        feat_dir = features + '_features' + '/' + data_label

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
                Tx = extract_features(data_dirs[k] + file + audio_ext, features=features, data_label=data_label, data_type=data_type, cached=True)
                print(Tx.shape)