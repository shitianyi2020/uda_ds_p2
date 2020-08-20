# Disaster Response Pipeline Project

### Project Overview
The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The final results are shown through a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Project Components
1. ETL Pipeline
'process_data.py', a data cleaning pipeline that:
   * Loads the messages and categories datasets 'disaster_messages.csv' & 'disaster_categories.csv'
   * Merges the two datasets
   * Cleans the data
   * Stores data in a SQLite database 'DisasterResponse.db'
2. ML Pipeline
'train_classifier.py', a machine learning pipeline that:
   * Loads data from the SQLite database
   * Splits the dataset into training and test sets
   * Builds a text processing and machine learning pipeline
   * Trains and tunes a model using GridSearchCV
   * Outputs results on the test set
   * Exports the final model as a pickle file 'classifier.pkl'
3. Flask Web App
'run.py', a process that:
   * Loads data and model
   * Builds a web framework by rendering 'go.html' & 'master.html'
   * Creates visualizations

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

