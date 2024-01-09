# from sklearn.mixture import GaussianMixture
from pycave.bayes import GaussianMixture
from pycave import set_logging_level
from random import sample
import pandas
import pickle
import math
import numpy as np
import os
import logging
from Asvspoof_dataset import PKL_dataset, open_pkl, gmm_custom_collate

set_logging_level(logging.INFO)


def train_gmm(data_label, features, train_keys, train_folders, audio_ext, dict_file, ncomp, feat_dir='features', init_only=False):

    print("Training GMM for {} data".format(data_label))

    # define directory to save the gmm model
    gmm_save_dir = '_'.join((dict_file, data_label))

    # path to the dataset
    path_to_dataset = os.path.join(feat_dir, features + '_features', 'train', data_label)

    # load pickle files
    total_files = os.listdir(path_to_dataset)

    # pickle all data into one file
    all_data_pickle_file = path_to_dataset + '.pkl'

    if not os.path.exists(all_data_pickle_file):

        for i, pkl_file in enumerate(total_files):

            pkl_data = open_pkl(os.path.join(path_to_dataset, pkl_file))

            print(i)

            if i == 0: 
                train_data = pkl_data
            else:
                train_data = np.vstack([train_data, pkl_data])

            # print(train_data.shape)
        
        with open(all_data_pickle_file, 'wb') as f:
            pickle.dump(train_data, f)

    else:

        train_data = open_pkl(all_data_pickle_file)

    
    # train the Gmm model on the data
    gmm = GaussianMixture(num_components=ncomp,
                            covariance_type='diag',
                            batch_size=64,
                            covariance_regularization=0.1,
                            init_strategy='kmeans++',
                            trainer_params=dict(accelerator='gpu', devices=1, max_epochs=100))

    history = gmm.fit(train_data)

    # save the trained model
    gmm.save(gmm_save_dir)

    return gmm