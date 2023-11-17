from nltk import pos_tag
from nltk.tokenize import word_tokenize

import nltk
import pandas as pd
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)



def count_pos_tag(text, tag):
    """
    Count occurrences of tag in sentence
    """
    pos_family = {
        'noun' : ['NN','NNS','NNP','NNPS'],
        'pron' : ['PRP','PRP$','WP','WP$'],
        'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
        'adj' :  ['JJ','JJR','JJS'],
        'adv' : ['RB','RBR','RBS','WRB']
    }
    tag_counter = 0
    sentence_pos_tags = pos_tag(word_tokenize(text))
    for s_pos_tag in sentence_pos_tags:
        s_tag = s_pos_tag[1]
        if s_tag[:2] in pos_family[tag]:
            tag_counter += 1
    return tag_counter


def create_syntactical_features(text, idx=0):
	"""
	Create syntactial and lexical features from a
	given text
	"""
	ret_dic = {}
	# create syntactical features
	ret_dic['noun_count'] = count_pos_tag(text, 'noun')
	ret_dic['verb_count'] = count_pos_tag(text, 'verb')
	ret_dic['adj_count'] = count_pos_tag(text, 'adj')
	ret_dic['adv_count'] = count_pos_tag(text, 'adv')
	ret_dic['pron_count'] = count_pos_tag(text, 'pron')
	# create lexical features
	ret_dic['chart_count'] = len(text)
	ret_dic['word_count'] = len(text.split())
	ret_dic['word_density'] = ret_dic['chart_count']/(ret_dic['word_count']+1)
	ret_dic['upper_case_word_count'] = len([wrd for wrd in text.split() if wrd.isupper()])
	ret_dic['title_word_count'] = len([wrd for wrd in text.split() if wrd.istitle()])
	return pd.DataFrame(ret_dic, index=[idx])


def remove_stops_digits(tokens, stopwords):
  """
  Remove stop words and digits from tokens, 
  putting them to lower case
  """
  return [token.lower() for token in tokens if token.lower() not in stopwords and not token.isdigit()]


def clean_segment(segment, remove_non_chars=True, remove_spaces=True):
    """
    Perform cleaning tasks on a given segment
    """
    if remove_non_chars:
      # remove non-alphabetic characters
      segment = re.sub(r"[^a-zA-Z]", " ", segment)
    if remove_spaces:
      # remove leading and trailing spaces
      segment = segment.strip()
      # remove extra spaces
      segment = ' '.join(segment.split())
    return segment


def preprocess_segments(segments):
  """
  Preprocess segments received as parameters
  """
  preprocess_text = []

  for segment in segments:
    # clean segment tokens
    segment = clean_segment(segment)
    # tokenize segment
    tokens = word_tokenize(segment)
    # put tokens to lower case
    tokens = [token.lower() for token in tokens]
    # add clean tokens to list
    preprocess_text.append(tokens)

  return preprocess_text
