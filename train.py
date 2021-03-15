
import argparse
import os
import numpy as np
import glob
import joblib
from azureml.core import Workspace, Dataset, Datastore, Run
from azureml.data.datapath import DataPath
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import preprocess
from azureml.core import Model
import nltk
nltk.download('wordnet')

run = Run.get_context()
ws = run.experiment.workspace

ds = Datastore.get(ws, datastore_name='stdatasciencelab')
print(ds)

df = Dataset.Tabular.from_json_lines_files(path = [(ds, '/train.jl')]).to_pandas_dataframe()
X_train = df['ingredients'].apply(preprocess)
Y_train = df['cuisine']

vectorizer2 = CountVectorizer()
X_train_vec = vectorizer2.fit_transform(X_train)

mnb = MultinomialNB()
mnb.fit(X_train_vec, Y_train)

X_test = vectorizer2.transform(X_train)
mnb.predict(X_test)
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=mnb, filename='outputs/mnb_model.pkl')

Model.register(workspace = ws,
    model_name='mnb-cooking',
    tags={'kind': 'demo'},
    model_path='./outputs/mnb_model.pkl',
    model_framework = Model.Framework.SCIKITLEARN,
    model_framework_version = sklearn.__version__)
