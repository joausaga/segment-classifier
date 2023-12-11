# Segment Classifier

Segment classifier proposes to leverage machine learning (ML) and natural language processing (NLP) techniques to build a text classifier that automatically identifies sections in job description documents.

The proposal aims to solve a multi-class classification problem in which the model takes a segment (or sentence) contained in a job description as input and produces 
as output one of seven possible sections that the segment belongs to.

## Dataset

The training dataset contains labeled job description segments (or sentences; either 
of both terms are used interchangeably). Each segment is assigned to one of seven
sections (labels), namely: *Job Responsibilities/Summary*, *Job Skills/Requirements*, 
*Other*, *About Company*, *Benefits*, *EOE/Diversity*, and *Job Title*. The 
distribution of segments by labels is uneven.

The data stored in `data/` is a `csv` file of `four columns` where each row 
contains the `job id`, representing the identifier of the job description document that the segment belongs to, the `segment index`, specifying the starting index 
of the segment in the job description, the `segment` text, and the `section label` 
that corresponds to the segment.

## Approaches

Two approaches were employed to solve the problem. On the one hand, classical
machine learning algorithms (also called "shallow methods," contrasting their deep learning counterparts) were used, while, on the other hand, a neural network was built. Next, the two approaches are presented in detail.

### 1. Classical machine learning algorithms

In this approach, features have been generated through two different strategies. First, features have been constructed by extracting lexical and syntactic information from segments. Second, features have been built by converting segments into vectors of tokens using the `doc2vec` technique.

Taking these features, a `Logistic Regression` and a `Suppor Vector Machine` have
been trained. Both algorithms have been reported to perform well on unbalanced, small, and textual datasets (e.g., [Text Classification with Extremely Small Datasets](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftowardsdatascience.com%2Ftext-classification-with-extremely-small-datasets-333d322caee2)). Models are trained using cross-validation and their performance is measured by calculating the `weighted F1`. This metric was selected because it provides an adequate compromise between acceptable coverage and the correct identification of sections.

More specifically, the process of building the models is composed of four steps:

1. Train algorithms using different combinations of hyperparameters;
2. Pre-select the best *logistic regression* and *support vector machine* models;
3. Evaluate the performance of the best *logistic regression* and *support vector machine* models on the test set;
4. Select the model that shows the best results on the test set.

According to the performance evaluation, the best classifier is a *support vector machine* with a *polynomial kernel* that reported a `weighted F1` of `0.76` on the test set. 

|              | precision | recall | f1-score |   |
|--------------|-----------|--------|----------| --|
| accuracy     |         |      |        | 0.76 |
| macro avg    | 0.71    | 0.68 | 0.69   |      |
| weighted avg | 0.76    | 0.76 | 0.76   |      |

The whole process followed in this approach is documented in the notebook `1_ml_segment_classifier-model_development.ipynb`.

### 2. Neural network classifier

A `long short-term memory (LSTM)` neural network architecture (a type of recurrent neural networks) has also been employed to approach the problem. The LSTM architecture
has reported to be extensively used for text classification (e.g., [A Survey on Text Classification: From Traditional to Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3495162)).

In this approach, segments have been converted to numerical vectors before feeding them into an embedding layer of the LSTM architecture. Cross entropy loss has been chosen as the loss function, and Adam as the optimizer. Several configurations of the LSTM architecture have been trained, including combinations of different hidden state sizes and number of layers. The most accurate configuration has been selected for evaluation on the test set. Also here, `F1` was chosen as the performance metric.

In particular, an LSTM with *256 hidden state size* and *2 layers* was the best-performing model, reporting a `weighted F1` score of `0.71` on the test set, as shown in the following table.


|              | precision | recall | f1-score |   |
|--------------|-----------|--------|----------| --|
| accuracy     |         |      |        | 0.72 |
| macro avg    | 0.62    | 0.61 | 0.61   |      |
| weighted avg | 0.71    | 0.72 | 0.71   |      |

The process followed to build the LSTM-based classifier is documented in the notebook `2_lstm_segment_classifier-model_development.ipynb`.

### 3. Approaches discussion

Contrary to what might be expected, classical machine learning techniques were shown to perform better than the neural network approach due to several reasons. First, the use of *doc2vec* might have provided a more insightful representation of the subtle structure of the text, resulting in better models. Second, augmenting the data with syntactical features might have helped learners better understand syntactic particularities in the text of each section. For example, segments in the *About Company* section might include more nouns or words in capital letters while segments in the *Job Responsability* section might have more verbs. These syntactic features capture these nuances in sections.

Also, as job descriptions are usually structured in sections from top to bottom, segments belonging to the same section might have similar segment indexes. This relationship might have been helpful for learners. Lastly, previous research (e.g., [A Comparison between Shallow and Deep
Architecture Classifiers on Small Dataset](https://ieeexplore.ieee.org/iel7/7839735/7863216/07863293.pdf?casa_token=KVQMeqi88vkAAAAA:zB9ZDZUPRfe6bgSNQVdnKDcdZ0Ph42M2oDNHsByRYtQDmGqns9pOXRGNv7GHFCmcpqsZwBRGXw)) found that classical machine learning techniques showed to perform better than neural network methods with small datasets.

### 4. Model selection

Given the explained results, the **support vector machine model** has been selected as the classifier to make predictions. This model not only outperforms the other alternatives (*logistic regression*, *LSTM*) but also shows better results than the `baseline model`, a *bidirectional LSTM*, which reports the results presented next.

|              | precision | recall | f1-score | |
|--------------|-----------|--------|----------|---|
| accuracy     |           |        |          | 0.63 |
| macro avg    | 0.5258    | 0.5784 | 0.5342   | |
| weighted avg | 0.6440    | 0.6283 | 0.6282   | |


## Classifier CLI usage

:warning: Before using the classifier, please make sure that you have previously followed all of the installation instructions.

The classifier is prepared to be used through the command line interface (CLI) by running the following command **from the directory of the repository**

`python app.py --input_fn data/jobs_test.csv`

Predictions results are stored in the `outputs/` directory under `prediction_output.csv`.

## Installation

1. Install `pyenv` to manage python installations. [Here](https://github.com/pyenv/pyenv#unixmacos) instructions for Unix-based systems;
2. Install Python 3.9.1, see [here](https://github.com/pyenv/pyenv#install-additional-python-versions)
3. Install `pyenv-virtualenv`, see [here](https://github.com/pyenv/pyenv-virtualenv#installation)
4. Create python virtual environment for the project, see [here](https://github.com/pyenv/pyenv-virtualenv#usage)
5. Clone the repository `git clone https://github.com/joausaga/segment-classifier.git`
6. Get into the repository directory `cd segment-classifier`
7. Install dependencies `pip install -r requirements.txt`

## Dependencies

1. Python 3.9.1
2. [PyEnv](https://github.com/pyenv/pyenv)
3. Packages indicated in `requirements.txt`

## Repository content

- `/data`: datasets (train and test)
- `/models`: models created during the solution implementation
- `/outputs`: outputs of the training processes and results of the predictions conducted on the test set
-  `1_ml_segment_classifier-model_development.ipynb`: methodology followed to build the logistic regression and support vector machine classifiers
-  `2_lstm_segment_classifier-model_development.ipynb`: methodology followed to develop the LSTM classifier
-  `app.py`: CLI application to use the segment classifier
-  `text_processor.py`: functions used to process the clean, transform, and process the text
-  `utils.py`: utilitarian functions employed for encoding/decoding of categorical variables and scaling of numerical features


