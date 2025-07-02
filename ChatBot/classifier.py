from sklearn.tree import DecisionTreeClassifier
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

with open('../PickleFiles/positive_sentiment.pickle', 'rb') as handle:
    positive_classifier = pickle.load(handle)

with open('../PickleFiles/aggressive_sentiment.pickle', 'rb') as handle:
    aggressive_classifier = pickle.load(handle)

embedding = model.encode("Hello!")
print(positive_classifier.predict([embedding]))
print(aggressive_classifier.predict([embedding]))
