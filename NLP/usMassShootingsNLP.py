import pandas as pd

from gensim.corpora.dictionary import Dictionary
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk


#returns a data frame
def loadFileToDF(path):
    df = pd.read_csv(path)
    return df

def listToString(wordLst):
    wordsStr = " "
    wordsStr = wordsStr.join(wordLst)
    return wordsStr


#return one massive string from a list of strings
def getStringAll(summaryLst):
    totalAllStr = ""
    for summary in summaryLst:
        totalAllStr += summary
    totalAllStr = " ".join(totalAllStr)
    return totalAllStr


#return one massive clean string for word analysis
def getCleanStrAll(summaryLst):
    totalStr = ""
    for summary in summaryLst:
        totalStr += summary
    cleanWordsAll = cleanWords(totalStr)
    cleanWordsAllStr = " ".join(cleanWordsAll)
    return cleanWordsAllStr

#returns a list
def getSentences(blobStr):
    sentences = sent_tokenize(blobStr)
    return sentences

#NER preprocessing get NER Chunked Sentences
def getNERposSent(blobStr):
    #NLTK Tokenize the string blob into sentences
    sentences = sent_tokenize(blobStr)
    # Tokenize each sentence into words: token_sentences
    token_sentences = [word_tokenize(sent) for sent in sentences]
    # Tag each tokenized sentence into parts of speech: pos_sentences
    pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]
    return(pos_sentences)

# clean the set of words by removing unecessary tokens, capitalizations,
# stop words, lower case, and replacing AR w AR-15. Returns a List of words.
def cleanWords(wordStr):
    # Tokenize and remove punctuation
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(wordStr)

    # Removing any numerical references. But might revisit this for a better approach
    alpha_only = [t for t in words if t.isalpha()]

    # replace AR with AR-15
    words2 = list(map(lambda x: x.replace('AR', 'AR-15'), alpha_only))

    #change to lower case and create a counter for frequency
    lower_words = [t.lower() for t in words2]

    #removing stop words
    sw = stopwords.words('english')
    sw.append('said')
    words3 = [word for word in lower_words if word.lower() not in sw]

    return(words3)


def getUniqueWords(wordLst):
    unique_tokens = set(wordLst)
    # print("# of tokens: " + str(len(tokens)))
    #print("# of unique tokens words: " + str(len(unique_tokens)))
    return(unique_tokens)

def getMostCommon(wordLst):
    lower_tokens = [t.lower() for t in wordLst]
    occurance = Counter(lower_tokens)
    return(occurance.most_common())


def getDict(wordLst):
    return(Dictionary([wordLst]))


#Creat Corpus, BOW, return corpus and ID for "shooting"
def getCorpusShtID(wordsLst, token):
    dictionary = getDict(wordsLst)
    # Select the id for "shooting": shooting_id
    token_id = dictionary.token2id.get(token)

    # Use shooting_id with the dictionary to print the word
    #print(dictionary.get(shooting_id))

    # Create a corpus
    corpus = [dictionary.doc2bow(wordsLst) for word in wordsLst]
    #print(corpus[0][:10])
    return(corpus, token_id)



