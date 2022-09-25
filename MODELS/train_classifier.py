import sys
import nltk 
nltk.download(['wordnet', 'punkt', 'stopwords'])
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC
import warnings
import pickle
import os
import re
import pickle

def load_data(database_filepath):
    """
    INPUT:
    database_filepath - 
    
    OUTPUT:
    X - messages (input variable) 
    y - categories of the messages (output variable)
    category_names - category name for y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    INPUT:
    text - raw text
    
    OUTPUT:
    clean_tokens - tokenized messages
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%     [0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(clf = AdaBoostClassifier()):
    """
    INPUT:
    clf - classifier model (If none is inputted, the function will use default 'AdaBoostClassifier' model) 
    
    OUTPUT:
    cv = ML model pipeline after performing grid search
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.5, 1.0],
        'clf__estimator__n_estimators':[10,20]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    modelo - ML model
    X_test - test messages
    y_test - categories for test messages
    category_names - category name for y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = np.squeeze(model.predict(X_test))
    print(classification_report(Y_test.values, Y_pred_test,           target_names=category_names))
    

def save_model(model, model_filepath):
    """
    INPUT:
    modelo - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,             test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()  