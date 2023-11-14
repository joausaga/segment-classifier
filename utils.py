from pickle import dump, load
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np
import os



def encode_labels(label_classes, label_values, save_dir=''):
  """
  Covert categorical labels into number
  """
  le = LabelEncoder()
  le.fit(label_classes)
  # save encoder to be used at testing
  np.save(os.path.join(save_dir, 'encoder_classes.npy'), le.classes_)
  return le.transform(label_values)


def decode_labels(num_labels, encoder_path):
  """
  Convert numerical labels to categories using
  a encoder located at encoder_dir
  """
  le = LabelEncoder()
  if os.path.isfile(encoder_path):
    le.classes_ = np.load(encoder_path)
    return list(le.inverse_transform(num_labels))
  else:
    raise Exception(f'Numbers cannot be transformed to categorical labels. '\
                    f'Encoder does not exist at {encoder_path}')


def scale_features(feature_values, scaler_dir):
  """
  Scale numerical features
  """
  scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
  if os.path.isfile(scaler_path):
    scaler = load(open(scaler_path, 'rb'))
  else:
    scaler = StandardScaler()
    scaler.fit(feature_values)
    dump(scaler, open(scaler_path, 'wb'))

  return scaler.transform(feature_values)
