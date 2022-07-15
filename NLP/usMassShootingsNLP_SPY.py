import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import usMassShootingsNLP

def getBOWmtx(wordStr):
    # Instantiate the English model: nlp
    nlp = spacy.load('en_core_web_sm')
    # Create a new document: doc
    doc = nlp(wordStr)

    # Print all of the found entities and their labels
    # Spacy adds additional categories that NLTK doesn't have like:
    # NORP, CARDINAL, MONEY, WORKOFART, LANGUAGE, EVENT
    for ent in doc.ents:
        print(ent.text, ent.label_)

    pos = [(token.text, token.pos_) for token in doc]

    # Create CountVectorizer object
    vectorizer = CountVectorizer()

    # Generate matrix of word vectors
    bow_matrix = vectorizer.fit_transform(pos.pop())

    # Print the shape of bow_matrix
    print("The shape of the BOW Matrix is: " + bow_matrix.shape)

    return(bow_matrix)