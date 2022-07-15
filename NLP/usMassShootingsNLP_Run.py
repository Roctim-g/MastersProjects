#!/usr/bin/python

import sys
import usMassShootingsNLP
import usMassShootingsNLP_Viz
import usMassShootingsNLP_SPY


"""
RUN SCRIPT for PROJECT 
"""


def main(argv):
    filename = sys.argv[1]
    # LOAD Data File
    #data_file = '/Users/aosanchez/Desktop/George Mason 580/Class Project/Data/50YRSSHOOTINGS/USMassShootings05242022_update0704.csv'
    data_file = filename

    df = usMassShootingsNLP.loadFileToDF(data_file)
    row = 0
    instanceStr = df.summary[row]

    # 1) Run word analsyis with single instance and then with all instances

    # print basics of text by instance and total corpus of data
    usMassShootingsNLP_Viz.printBasicInfo(df.summary, row)

    # Clean words for single and then all instances of type List and String
    # For use in different methods
    cleanWordsLst = usMassShootingsNLP.cleanWords(instanceStr)
    cleanWordsStr = usMassShootingsNLP.listToString(cleanWordsLst)
    totalCleanWordsStr = usMassShootingsNLP.getCleanStrAll(df.summary)
    totalCleanWordsLst = usMassShootingsNLP.cleanWords(totalCleanWordsStr)

    # Print a word cloud with one instance and one with all instances on the map of US
    usMassShootingsNLP_Viz.showWordCloud(cleanWordsStr)
    usMassShootingsNLP_Viz.showWordCloudUS(totalCleanWordsStr)

    # return the most unique words and print the list with frequencies
    print("Number of unique words in list: " + str(len(usMassShootingsNLP.getUniqueWords(totalCleanWordsLst))))
    usMassShootingsNLP_Viz.printMostCommon(totalCleanWordsLst, 20)

    # Perform NER visualizations with Noun Phrase Chunking
    sentence_num = 2
    sentences = usMassShootingsNLP.getSentences(instanceStr)
    pos_sentences = usMassShootingsNLP.getNERposSent(instanceStr)
    # Prints out NER list, Test for stems of the tree with NE tags
    usMassShootingsNLP_Viz.printNERChunk(pos_sentences)
    # display Pie Chart dissecting the POS
    usMassShootingsNLP_Viz.showDict(pos_sentences)

    print(sentences[sentence_num])
    usMassShootingsNLP_Viz.drawNPChunk(pos_sentences[sentence_num])

    # Run Corpus test and printing out Tfid weights and frequencies
    """
    corpus, id = usMassShootingsNLP.getCorpusShtID(cleanWordsLst, 'shooting')
    print(id)
    usMassShootingsNLP_Viz.showTfidWeights(corpus)
    usMassShootingsNLP_Viz.getBOW(corpus, cleanWordsLst, 50)
    """

    # Using BOW with Spacy, redundent to some extents for the project we're working on.

    # bowMatrix = usMassShootingsNLP_SPY.getBOWmtx(instanceStr)


if __name__ == "__main__":
    main(sys.argv[1:])




