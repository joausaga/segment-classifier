import click
import joblib
import json
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from gensim.models.doc2vec import Doc2Vec
from text_processor import *
from utils import *


def get_prediction_class(prediction, encoder_file_path):
    class_nums = list(range(0,7))
    class_names = decode_labels(class_nums, encoder_file_path)
    for class_num, class_name in zip(class_nums, class_names):
        if class_num == prediction:
            return class_name


@click.command()
@click.option('--input_fn', '-ifn', default='')
@click.option('--outputs_dir', '-odir', default='outputs')
@click.option('--output_fn', '-ofn', default='prediction_output.csv')
@click.option('--training_outputs_fn', '-tofn', default='output_training.json')
@click.option('--models_dir', '-mdir', default='models')
@click.option('--classifier_name', '-clf', default='SVM')
@click.option('--doc2vec_model_fn', '-d2vm', default='doc2vec.model')
@click.option('--encoder_fn', '-efn', default='encoder_classes.npy')
@click.option('--verbose', '-v', is_flag=True, default=True)
def main(input_fn, outputs_dir, output_fn, training_outputs_fn, models_dir, 
         classifier_name, doc2vec_model_fn, encoder_fn, verbose):
    output_file_path = os.path.join(outputs_dir, output_fn)
    if input_fn:
        print('-'*10)
        print(f'Identifying sections of segments in {input_fn} using a ' \
              f'{classifier_name} classifier, please wait...')
        print('-'*10)
        # 0. read input file
        input_df = pd.read_csv(input_fn)
        # 1. create syntactial features and concatenate with segment index
        if verbose:
            print('[1/7] Creating syntactical features...')
        syn_features_df = pd.DataFrame()
        for idx, row in input_df.iterrows():
            syn_features_df = pd.concat(
                [syn_features_df, create_syntactical_features(row['segment'], idx)]
            )
        new_input_df = pd.concat([input_df, syn_features_df.reindex(input_df.index)], 
                                 axis=1)
        # 2. preprocess segment texts
        if verbose:
            print('[2/7] Preprocessing segment texts...')
        processed_segs = preprocess_segments(new_input_df['segment'])
        # 3. scale numerical features
        if verbose:
            print('[3/7] Scaling numerical features...')
        features_to_scale = ['segment_index'] 
        features_to_scale.extend(list(syn_features_df.columns))
        array_num_features = np.array(new_input_df.loc[:,features_to_scale])
        scaled_features_array = scale_features(array_num_features, models_dir)
        scaled_num_features_df = pd.DataFrame(scaled_features_array, 
                                 columns=features_to_scale)
        # 4. generate doc from segment tokens
        if verbose:
            print('[4/7] Generating doc embeddings for segments...')
        doc2vec_model_file_path = os.path.join(models_dir, doc2vec_model_fn)
        doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)
        doc_vectors =  [doc2vec_model.infer_vector(seg_tokens) 
                        for seg_tokens in processed_segs]
        # 5. concatenate doc vector2 with numerical features
        features = np.concatenate((doc_vectors, scaled_num_features_df), axis=1)
        # 6. load model
        if verbose:
            print(f'[5/7] Loading {classifier_name} model classifier...')
        classifier = None
        training_outputs_file_path = os.path.join(outputs_dir, training_outputs_fn)
        with open(training_outputs_file_path, 'r') as f:
            training_outputs = json.load(f)
            for training in training_outputs:
                if training['algorithm'] == classifier_name:
                    model_file_path = training['model_file_path']
                    classifier = joblib.load(model_file_path)['model']
        if classifier:
            # 7. make predictions
            if verbose:
                print('[6/7] Making predictions...')
            predictions = classifier.predict(features)
            # 8. convert prediction numbers to their corresponding classes
            encoder_file_path = os.path.join(models_dir, encoder_fn)
            predictions_df = pd.DataFrame(predictions, columns=['prediction'])
            predictions_df['prediction_class'] = \
                predictions_df['prediction'].apply(
                    lambda x: get_prediction_class(x, encoder_file_path))
            # 9. save predictions
            if verbose:
                print('[7/7] Saving predictions...')
            input_df = pd.concat([input_df, predictions_df['prediction_class']], 
                                 axis=1)
            input_df.to_csv(output_file_path)
            print('-'*10)
            print(f'Classification has finished, results can be accessed at ' \
                  f'{output_file_path}')
            print()
        else:
            raise Exception('Classifier could not be loaded')
    else:
        raise Exception('Please provide an input file name, including the full '
                        'path to the file')


if __name__ == "__main__":
    main()