from sklearn.tree import DecisionTreeClassifier
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def ChatBot():
    def __init__(self):
        pass

class ClassifierModelBuilder():
    def __init__(self):
        pass

class ChatBotBuilder:
    def __init__(self):
        self.is_reset_classifiers = False
        self.classifier_builder = ClassifierModelBuilder()
        self.embedding_method = lambda x: model.encode(x)

    def set_embedding_method(self, embedding_method):
        self.embedding_method = embedding_method

    def set_llm_standard_answer_method(self, standard_answer_method):
        self.standard_answer_method = standard_answer_method

    def set_llm_negative_answer_method(self, negative_answer_method):
        self.negative_answer_method = negative_answer_method

    def set_llm_aggressive_answer_method(self, aggressive_answer_method):
        self.aggressive_answer_method = aggressive_answer_method

    def reset_classifiers(self):
        self.is_reset_classifiers = True

    def build_chatbot(self):
        pass

    def build_negative_classifier(self):
        pass