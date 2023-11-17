# Segment Classifier

Segment classifier proposes to leverage machine learning (ML) and natural language processing (NLP) techniques to build a text classifier that automatically identifies sections in job description documents.

The proposal aims to solve a multi-class classification problem in which the model takes a segment (or sentence) contained in a job description as input and produces 
as output one of seven possible sections that the segment belongs to.

## Dataset

The training dataset contains labeled job description segment (or sentences; either 
of both terms are used interchangeably). Each segment is assigned to one of seven
sections (labels), namely: `Job Responsibilities/Summary`, `Job Skills/Requirements`, 
`Other`, `About Company`, `Benefits`, `EOE/Diversity`, and `Job Title`. The 
distribution of segments by labels is uneven.

The data stored in `data/` is a `csv` file of `four columns` where each row 
contains the `job id`, representing the identifier of the job description document that the segment belongs to, the `segment index`, specifying the starting index 
of the segment in the job description, the `segment` text, and the `section label` 
that corresponds to the segment.

## Approaches

Two approaches were employed to solve the problem. On the one hand, classical
machine learning algorithms were used, while, on the other hand, a neural network
was built. Next, the two approaches are presented in details.

### Classical machine learning algorithms

In this approach features have been generated through two different strategies. First, features have been constructed by extracting some lexical and syntactic information from segments. Second, features have been built by converting  segments into vectors of tokens using the `doc2vec` technique.

Taking these features a `Logistic Regression` and a `Suppor Vector Machine` have
been trained. Both algorithms have been reported to perform well on unbalanced, small, and textual datasets (e.g., [Text Classification with Extremely Small Datasets](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftowardsdatascience.com%2Ftext-classification-with-extremely-small-datasets-333d322caee2)). Models are trained using the cross-validation approach. The performance metric to be optimized is weighted `F1` because it provides an adequate compromise between acceptable coverage and the correct identification of segments.

More specifically, the process of bulding the model is composed of four steps:

1. Train algorithms using different combinations of hyperparameters;
2. Pre-select the best `logistic regression` and `support vector machine` models;
3. Evalute the performance of the best `logistic regression` and `support vector machine` models on the test set;
4. Select the model that shows the best results on the test set.

According to the evaluation of the test set, the best classifier is a `support vector machine` with a `polynomial kernel` that showed a `F1` score of `0.78` on the test set. The whole process followed to build the model has been documented in the notebook `1_ml_segment_classifier-model_development.ipynb`.

### Neural 

## Baseline classifier

A bidirectional LSTM with a softmax layers was used for comparison. This classifier
reports to have an `accuracy` of `0.63` and the following results for `pression`, 
`recall`, and `f1`.

|              | precision | recall | f1-score |
|--------------|-----------|--------|----------|
| macro avg    | 0.5258    | 0.5784 | 0.5342   |
| weighted avg | 0.6440    | 0.6283 | 0.6282   |

Both of the presented approaches beat the baseline classifier.

