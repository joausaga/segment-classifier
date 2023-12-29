import click
import os
import pandas as pd
import torch
import warnings

warnings.filterwarnings('ignore')

from transformers import BertTokenizer
from utils import *


def get_prediction_class(prediction, encoder_file_path):
    class_nums = list(range(0,7))
    class_names = decode_labels(class_nums, encoder_file_path)
    for class_num, class_name in zip(class_nums, class_names):
        if class_num == prediction:
            return class_name


def segment_preprocessing(segment, tokenizer, max_length=50):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens 
    should considered by the model (return_attention_mask = True).
  '''
  encoded_segment = tokenizer.encode_plus(
    segment,
    add_special_tokens = True,
    max_length = max_length,
    pad_to_max_length = True,
    return_attention_mask = True,
    return_tensors = 'pt',
    truncation = True
  )
  return encoded_segment


def make_prediction(model, ids, attention_mask, device):
  with torch.no_grad():
    model_output = model(
        ids.to(device),
        token_type_ids=None,
        attention_mask=attention_mask.to(device)
    )
  pred = np.argmax(model_output.logits.cpu().numpy()).flatten().item()
  return pred


@click.command()
@click.option('--input_fn', '-ifn', default='')
@click.option('--outputs_dir', '-odir', default='outputs')
@click.option('--output_fn', '-ofn', default='prediction_output.csv')
@click.option('--training_outputs_fn', '-tofn', default='output_ml_training.json')
@click.option('--models_dir', '-mdir', default='models')
@click.option('--classifier_name', '-clf', default='BERT')
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
        # 1. preprocess segment texts
        if verbose:
            print('[1/4] Preprocessing segment texts...')
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', 
            do_lower_case=True
        )    
        token_ids = []
        attention_masks = []
        max_length = 200
        for segment in input_df.segment:
            encoding_dict = segment_preprocessing(segment, tokenizer, max_length)
            token_ids.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_ids, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        # 2. load model
        if verbose:
            print(f'[2/4] Loading {classifier_name} model classifier...')
        model_name = f'bert_model.pth'
        model_file_path = os.path.join(models_dir, model_name)
        classifier = torch.load(model_file_path)
        if classifier:
            # 3. make predictions
            if verbose:
                print('[3/4] Making predictions...')
            if torch.cuda.is_available():
                print('GPU available and will be used for training.')
                device = torch.device('cuda')
            else:
                print('Only CPU available.')
                device = torch.device('cpu')
            predictions = make_prediction(
                classifier, 
                token_id, 
                attention_masks, 
                device
            )
            # 8. convert prediction numbers to their corresponding classes
            encoder_file_path = os.path.join(models_dir, encoder_fn)
            predictions_df = pd.DataFrame(predictions, columns=['prediction'])
            predictions_df['prediction_class'] = \
                predictions_df['prediction'].apply(
                    lambda x: get_prediction_class(x, encoder_file_path))
            # 9. save predictions
            if verbose:
                print('[4/4] Saving predictions...')
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