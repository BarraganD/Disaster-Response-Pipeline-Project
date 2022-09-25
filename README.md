# Disaster-Response-Pipeline-Project
DataScientist
Proyect 2 Disaster-Response-Pipeline-Project

**Installation**

The languages, programs that were used to carry out this project were: HTML y Python, Some of the python libraries are: pandas, numpy, re, pickle, nltk, json, plotly, sklearn, sqlalchemy, sys. However, the details of all the libraries and packages are found in the folders.

**Project Motivation**

The motivation for this project is to become an excellent professional as a data scientist and more by putting knowledge into practice on real-life cases, where knowledge and recognition are available for any solution to humanity's problems.

# File Description

**Data**
This Folder contains:
disaster_messages.csv - real messages sent during disaster events
disaster_categories.csv - categories of the messages
process_data.py  
ETL Pipeline Preparation.ipynb - Jupyter Notebook for prepare ETL pipeline
ML Pipeline Preparation.ipynb - Jupyter Notebook for prepare ML pipeline
DisasterResponse.db - cleaned data in SQlite

**Models**
This Folder contains:
train_classifier.py - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use
classifier.pkl - pickle file contains trained model

**APP**
This Folder contains:
run.py - python script to play web application.
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

# Instructions

1.Run the following commands in the project's root directory to set up your database and model.

-o *To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
-o *To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2.Run the following command in the app's directory to run your web app. python run.py
