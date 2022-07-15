import nltk
import usMassShootingsNLP
from gensim.models.tfidfmodel import TfidfModel
from collections import defaultdict
from nltk.corpus import stopwords
# Create a Dictionary from the words: dictionary
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
import re, pprint
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
from PIL import Image
import itertools

#Print Word List, String Basics Info

def printBasicInfo(stringLst, row):
    print("Number of instances in DataSet List of Strings: " + str(len(stringLst)))
    words = word_tokenize(stringLst[row])
    print("Number of words in Summary instance: " + str(len(words)))
    cleanwords = usMassShootingsNLP.getCleanStrAll(stringLst[row])
    print("Number of clean words in Summary instance: " + str(len(cleanwords)))

    totalwords = usMassShootingsNLP.getStringAll(stringLst)
    totalCleanwords = usMassShootingsNLP.getCleanStrAll(stringLst)
    print("Total number of words in Summary column: " + str(len(totalwords)))
    print("Total number of clean words in Summary column: " + str(len(totalCleanwords)))

    print("Number of sentences in Summary instance: ", len(sent_tokenize(stringLst[row])))
    totalstrlist = usMassShootingsNLP.listToString(stringLst)
    print("Number of sentences in Summary column: ", len(sent_tokenize(totalstrlist)))


# Returns most common words and plots the frequency
def printMostCommon(wordStr, num):
    freq_dist = nltk.FreqDist(wordStr)
    freq_dist.plot(num)
    print(freq_dist.most_common(num))

# Noun Phrase Chunking with RE, Tagging, Exploring Text Corpus
def drawNPChunk(sentence):
    #grammar = "NP: {<DT>?<JJ>*<NN>}"
    grammar = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sentence)
    #print(result)
    result.draw()

# Applying NER with NLTK
def printNERChunk(pos_sentences):
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
    # Test for stems of the tree with 'NE' tags
    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, "label") and chunk.label() == "NE":
                print(chunk)

#Show value Dict
def showDict(pos_sentences):
    #Create a defaultdict called ner_categories, with the default type set to int
    ner_categories = defaultdict(int)

    # Changing to non binary
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=False)

    #Changing to non binary
    # Create the nested for loop
    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, 'label'):
                ner_categories[chunk.label()] += 1

    # Create a list from the dictionary keys for the chart labels: labels
    labels = list(ner_categories.keys())

    # Create a list of the values: values
    values = [ner_categories.get(v) for v in labels]

    # Create the pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

    # Display the chart
    plt.show()

#Using Tf-idf to determine new significant terms for your corpus by applying gensim's tf-idf.
def showTfidWeights(corpus):
    # Save teh fifth document: doc
    doc = corpus[5]
    # Create a new TfidfModel using the corpus: tfidf
    tfidf = TfidfModel(corpus)
    # Calculate the tfidf weights of doc: tfidf_weights
    tfidf_weights = tfidf[doc]
    # Print the first five weights
    print(tfidf_weights[:100])


#Create BOW from document
def getBOW(corpus, wordLst, num):
    doc = corpus[num]
    dictionary = usMassShootingsNLP.getDict(wordLst)
    # Sort the doc for frequency: bow_doc
    bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

    # Print the top 5 words of the document alongside the count
    for word_id, word_count in bow_doc[:num]:
        print(dictionary.get(word_id), word_count)

    # Create the defaultdict: total_word_count
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count


def showWordCloud(wordStr):
    # Reusing stop word list and updating it
    sw = stopwords.words('english')
    # Adding "said" to the list of unecessary words
    sw.append('said')

    wordcloud = WordCloud(stopwords=sw, background_color="white").generate(wordStr)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # store to file
    plt.savefig("/Users/aosanchez/Desktop/George Mason 580/Class Project/Data/50YRSSHOOTINGS/us_shoot.png", format="png")

    plt.show()

def showWordCloudUS(wordStr):
    # Generate a word cloud image
    mask = np.array(Image.open("/Users/aosanchez/Desktop/George Mason 580/Class Project/Data/50YRSSHOOTINGS/us.png"))
    sw = stopwords.words('english')
    sw.append('said')
    wordcloud_usa = WordCloud(stopwords=sw, background_color="white", mode="RGBA", max_words=1000,
                              mask=mask).generate(wordStr)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[12, 12])
    plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
    #plt.imshow(wordcloud_usa, interpolation="bilinear")
    plt.axis("off")

    # store to file
    plt.savefig("/Users/aosanchez/Desktop/George Mason 580/Class Project/Data/50YRSSHOOTINGS/us_shootings.png", format="png")

    plt.show()