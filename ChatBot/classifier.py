from sklearn.tree import DecisionTreeClassifier
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

with open('../PickleFiles/positive_sentiment.pickle', 'rb') as positive_pickle:
    positive_classifier = pickle.load(positive_pickle)

with open('../PickleFiles/aggressive_sentiment.pickle', 'rb') as aggressive_pickle:
    aggressive_classifier = pickle.load(aggressive_pickle)

with open('../PickleFiles/negative_sentiment.pickle', 'rb') as negative_pickle:
    negative_classifier = pickle.load(negative_pickle)

embedding = model.encode("Hello!")
print(positive_classifier.predict([embedding]))
print(aggressive_classifier.predict([embedding]))
print(negative_classifier.predict([embedding]))
