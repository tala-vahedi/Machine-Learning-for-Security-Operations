# Script Purpose: Using NLTK Naive Bayes Classifier in Python
# Script Version: 1.0 
# Script Author:  Tala Vahedi, University of Arizona

# Script Revision History:
# Version 1.0 Oct 13, 2021, Python 3.x

'''
Using the provided GenderID.py script as a baseline.  Expand the script to include additional 
features to distinguish between male and female names.  To do this you will be modifying the 
genderFeatures function.  Make sure to examine genderIDv3 below for some initial hints.

Submit two files:
1) Your python script
2) A short document that includes the output of your script and your analysis.  Note prepare a 
test set of names to run through the classifier and calculate the accuracy.

'''

# Psuedo Constants
SCRIPT_NAME    = "Script: Using NLTK Naive Bayes Classifier in Python"
SCRIPT_VERSION = "Version 1.0"
SCRIPT_AUTHOR  = "Author: Tala Vahedi"

from nltk.corpus import names
from nltk import NaiveBayesClassifier,classify, accuracy
import random
import collections

print("Simple Gender Name Identifier using a NaiveBayesClassifier\n")

def genderFeatures(theName): 
    # getting the name and performing case foldings
    theName = theName.lower()
    # getting length of name
    theNameLen = len(theName)
    # how many times did the most unique character appear in name
    theUniqueCharLen = len(set(theName))
    # getting the most commong character in the name
    theMostCommonChar = collections.Counter(theName).most_common(1)[0]
    # list of vowels
    VOWELS = 'aeiouy'
    # list of consonants
    CONSONANTS = 'bcdfghjklmnpqrstvxz'
    # initializing count variables for both vowels and consonants
    vowelCnt = 0
    consonantCnt = 0
    # iterating through the name
    for eachLetter in theName:
        # if the letter is in the vowerl or consonant, increment count vars
        if eachLetter in VOWELS:
            vowelCnt += 1
        if eachLetter in CONSONANTS:
            consonantCnt += 1
        
    # establishing a dictionary for the features
    features = {}
    # giving each feature its associated variable 
    features['name'] = theName                                                      # getting the name
    features['nameLen'] = theNameLen                                                # getting name length
    features['uniqueCharLen'] = theUniqueCharLen                                    # getting count of most common letter
    features['mostCommonChar'] = theMostCommonChar                                  # getting most common letter
    features['charRatio'] = round(theUniqueCharLen/theNameLen)                      # getting ratio
    features['vowelPercent'] = round(vowelCnt/theNameLen, 2)                        # getting percentage
    features['consonantPercent'] = round(consonantCnt/theNameLen, 2)                # getting percentage
    features['vowelConsonantRatio'] = round(vowelCnt+consonantCnt/theNameLen, 2)    # getting ratio
    features['lastLetter'] = theName[-1]                                            # getting last letter
    features['firstLetter'] = theName[0]                                            # getting first letter
    # returning the features to use it again later
    return features

if __name__ == '__main__':
    # Print Basic Script Information
    print()
    print(SCRIPT_NAME)
    print(SCRIPT_VERSION)
    print(SCRIPT_AUTHOR)
    print() 

    # Collect the names label from the NTLK Corpus names
    nameLabels = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(nameLabels)

    # randomly grabbing features 
    sampleSize = len(nameLabels)
    # leaving 80% of data for training
    # the train test ratio was tweaked to see 
    # which ratio performed the best
    trainingSize = int(sampleSize * .80)
    # subtracting the rest of the data for testing
    testSize = sampleSize - trainingSize

    # Create a Feature set using the last Letter of each name
    trainingFeatureSet = [(genderFeatures(n), gender) for (n, gender) in nameLabels[0:trainingSize]]
    testingFeatureSet  = [(genderFeatures(n), gender) for (n, gender) in nameLabels[trainingSize:]]

    # Create a NaiveBayes Gender Classifer from the Training Set
    genderClassifer = NaiveBayesClassifier.train(trainingFeatureSet)

    # printing out the training and testing test accuracy
    print('TrainSet Accuracy: ',classify.accuracy(genderClassifer, trainingFeatureSet)) 
    print('TestSet  Accuracy: ',classify.accuracy(genderClassifer, testingFeatureSet),"\n") 
    # printing the most informative features in the classification process
    genderClassifer.show_most_informative_features(20)
    print("\nScript Ended")