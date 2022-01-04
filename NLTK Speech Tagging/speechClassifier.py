# Script Purpose: Using NLTK speech tagging in Python
# Script Version: 1.0 
# Script Author:  Tala Vahedi, University of Arizona

# Script Revision History:
# Version 1.0 Oct 5, 2021, Python 3.x

# Psuedo Constants
SCRIPT_NAME    = "Script: Using NLTK speech tagging in Python"
SCRIPT_VERSION = "Version 1.0"
SCRIPT_AUTHOR  = "Author: Tala Vahedi"

# import python standard libraries
from collections import Counter
import re
import csv 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


#import python 3rd party libs
from nltk import word_tokenize, pos_tag
from sklearn import metrics
from itertools import chain
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def PreProcess():
    # Preapre a STOP_WORD List
    with open("STOP_WORDS.txt", 'r') as stops:
        stopWords = stops.read()
    STOP_WORDS = stopWords.split()
    # two lists to hold the cadidate and candidate repsonse column values
    candidate = []
    candidateResponse = []
    # opening the debate file with cp1252 encoding to make readable
    with open("debate2clean.txt", 'r', encoding='cp1252') as debateFile:
        # iterating thru each line
        for eachLine in debateFile:
            # making all text lower case
            eachLine = eachLine.lower()
            # grab all text before the first comma in text file (e.g., the candidate name)
            c = re.compile('^(.+?),').findall(eachLine)[0]
            # removing the moderator lines
            if "joe biden" in c or "michael bennet" in c or "michael bloomberg" in c or "cory booker" in c or "pete buttigieg" in c or "julian castro" in c or "john delaney" in c or "tulsi gabbard" in c or "amy klobuchar" in c or "deval patrick" in c or "bernie sanders" in c or "tom steyer" in c or "elizabeth warren" in c or "marianne williamson" in c or "andrew yang" in c:
                # appending the candidates to the candidate list
                candidate.append(c)
            # grabing all text after the first comma in text file (e.g., the candidate response)
            cr = re.compile(',.*$').findall(eachLine)[0]
            # Remove any punctation or other characters from the speach
            cr = re.sub("[^a-zA-Z0-9]+", ' ', cr).split()
            # removing stop words from the text based on the stop words text file
            resultWords  = [word for word in cr if word.lower() not in STOP_WORDS]
            # creating a new column with cleaned text
            crResult = ' '.join(resultWords)
            # removing instances that are blank or null
            if not crResult:
                pass
            else:
                candidateResponse.append(crResult)
    
    # creating a csv with two columns: candidate name and the candidate response
    with open('clean.csv', 'w', encoding='cp1252') as f:
        writer = csv.writer(f)
        # csv file contains two columns, candidate and candidate response
        writer.writerows(zip(candidate, candidateResponse))

# a counter function that counts all the tags within each row 
def CountTags(taggedResponses):
    # initializing a dictionary to keep track of all tags and their counts
    count = {}
    for word, tag in taggedResponses:
        if tag in count:
            # adding one to the tag count if tag is repeated
            count[tag] += 1
        else:
            # else keeping it at 1
            count[tag] = 1
    return(count)


if __name__ == '__main__':
    # Print Basic Script Information
    print()
    print(SCRIPT_NAME)
    print(SCRIPT_VERSION)
    print(SCRIPT_AUTHOR)
    print() 
    PreProcess()
    
    # reading the pandas csv created in the PreProcess() function
    df = pd.read_csv('clean.csv', encoding='cp1252')
    # giving the csv a column names
    df.columns =['candidate', 'candidateResponse']
    # tokenizing and tagging each row in candidate response dataframe
    tags = df['candidateResponse'].apply(word_tokenize).apply(pos_tag)
    # creating a new dataframe
    tags = pd.DataFrame(tags)
    # applying the counts function on the candidate responses in column 
    tags['tagCounts'] = tags['candidateResponse'].map(CountTags)
    # getting a master list of all tags found in the candidate responses
    tagSet = list(set([tag for tags in tags['tagCounts'] for tag in tags]))
    # creating a new dataframe
    df2 = df
    # iterating through each tag in the master list and creating a column for that tag
    for tag in tagSet:
        # mapping the count for each tag to the associated column name, else putting 0
        df2[tag] = tags['tagCounts'].map(lambda x: x.get(tag, 0))
    # dropping the candidate response column since it is no longer needed
    df2 = df2.drop(['candidateResponse'], axis=1)

    # split the data into features and target variables
    candidate = df2.iloc[:, 0].values
    features = df2.loc[:, df2.columns != 'candidate']

    # create a knn classifier
    classifier = KNeighborsClassifier(n_neighbors=1)

    # fit the model with data
    classifier.fit(features, candidate)
    # predict the response values for the observations in X
    candidatePredict = classifier.predict(features)
    for feature in features:
        for canP in candidatePredict:
            print("Features: ",feature, " Prediction: ", canP)
    print("KNN Accuracy: ", metrics.accuracy_score(candidate, candidatePredict))