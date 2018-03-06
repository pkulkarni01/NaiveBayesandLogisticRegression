import os
import codecs
import math
from os.path import isfile, join

trainDir = "//Users//prathik//Desktop//MachineLearning//NaiveBayesAndLogisticRegression//train"
testDir = "//Users//prathik//Desktop//MachineLearning//NaiveBayesAndLogisticRegression//test"

prior = {}
probWordsNotSpam = {}
probWordsSpam = {}
totalwordlist = {}


def extractWordList(trainDir):
    allFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(trainDir) for f in filenames]
    allFiles.pop(0)
    for file in allFiles:
        with codecs.open(file, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                for word in line.split():
                    if word in totalwordlist:
                        frequency = totalwordlist[word]
                        frequency += 1
                    else:
                        frequency = 1
                    totalwordlist[word] = frequency

    return [totalwordlist, allFiles]


def extractWordListClass(trainDir, thisclass):
    wordlist = {}
    classFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(trainDir) for f in filenames if
                  f.endswith(thisclass + ".txt")]
    for file in classFiles:
        with codecs.open(file, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                for word in line.split():
                    if word in wordlist:
                        frequency = wordlist[word]
                        frequency += 1
                    else:
                        frequency = 1
                    wordlist[word] = frequency
    return wordlist


def trainNaiveBayes(trainDir):
    if os.path.isdir(trainDir):
        returnedList = extractWordList(trainDir)
        wordList = returnedList[0]
        print(wordList)
        countallFiles = len(returnedList[1])
        nooffilesinHamClass = len([os.path.join(dp, f) for dp, dn, filenames in os.walk(trainDir) for f in filenames if
                                   f.endswith("ham.txt")])
        prior["Not spam"] = nooffilesinHamClass / countallFiles
        wordsinHamClass = extractWordListClass(trainDir, "ham")
        totalFrequency = 0
        for key in wordsinHamClass:
            totalFrequency += wordsinHamClass[key]
        laplacesmoothingDenominator = totalFrequency + len(wordsinHamClass)
        for key in wordsinHamClass:
            numerator = wordsinHamClass[key] + 1
            probabilty = numerator / laplacesmoothingDenominator
            probWordsNotSpam[key] = probabilty
        print(wordsinHamClass)
        print(probWordsNotSpam)
        nooffilesinSpamClass = len([os.path.join(dp, f) for dp, dn, filenames in os.walk(trainDir) for f in filenames if
                                    f.endswith("spam.txt")])
        prior["spam"] = nooffilesinSpamClass / countallFiles
        wordsinSpamClass = extractWordListClass(trainDir, "spam")
        totalFrequency = 0
        for key in wordsinSpamClass:
            totalFrequency += wordsinSpamClass[key]
        laplacesmoothingDenominator = totalFrequency + len(wordsinSpamClass)
        for key in wordsinSpamClass:
            numerator = wordsinSpamClass[key] + 1
            probabilty = numerator / laplacesmoothingDenominator
            probWordsSpam[key] = probabilty
        print(wordsinSpamClass)
        print(probWordsSpam)
        print(prior)


def applyNaiveBayes(file):
    wordsinFile = {}
    with codecs.open(file, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            for word in line.split():
                if word in wordsinFile:
                    frequency = wordsinFile[word]
                    frequency += 1
                else:
                    frequency = 1
                wordsinFile[word] = frequency
    hamScore = math.log(prior['Not spam'])
    for word in wordsinFile:
        if word in probWordsNotSpam:
            probability = probWordsNotSpam[word]
            probability = math.log(probability)
        else:
            prob = 1 / (len(totalwordlist) + len(probWordsNotSpam))
            probability = math.log(prob)
        hamScore += probability
    # print(hamScore)
    spamScore = math.log(prior['spam'])
    for word in wordsinFile:
        if word in probWordsSpam:
            probability = probWordsSpam[word]
            probability = math.log(probability)
        else:
            prob = 1 / (len(totalwordlist) + len(probWordsSpam))
            probability = math.log(prob)
        spamScore += probability
    if hamScore > spamScore:
        return "not spam"
    else:
        return "spam"


def calcAccuracy(file):
    allFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(file) for f in filenames]
    allFiles.pop(0)
    print(len(allFiles))
    accuracy = 0
    for file in allFiles:
        predictedClass = applyNaiveBayes(file)
        if file.endswith("ham.txt"):
            originalClass = "not spam"
        else:
            originalClass = "spam"
        if predictedClass == originalClass:
            accuracy += 1

        print(file + " :" + predictedClass)
    accuracy = (accuracy / len(allFiles)) * 100
    return accuracy


trainNaiveBayes(trainDir)
trainAccuracy = calcAccuracy(trainDir)
print(trainAccuracy)
testAccuracy = calcAccuracy(testDir)
print(testAccuracy)
