from sklearn.tree import DecisionTreeClassifier
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np

default_model = SentenceTransformer("BAAI/bge-m3")
default_llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)
default_chatbot_response_function = lambda x: default_llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": x},
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.7,
    )

class ChatBot:
    def __init__(self, embedding_function, response_functions: dict, negative_classifier_function):
        self.embedding_function = embedding_function
        self.response_functions = response_functions
        self.negative_classifier_function = negative_classifier_function

    def query(self, query):
        if self.is_negative_query(query):
            return "I just wanted to check in to make sure you were okay"

        return self.response_functions["standard"](query)

    def is_negative_query(self, query):
        embedding = self.embedding_function(query.split("."))
        classifications = self.negative_classifier_function(embedding)
        print(classifications)
        return np.max(classifications)

def check_file_exists(relative_file_path):
    pass

def construct_classifier(classifier_identifier, is_reset_classifiers):
    default_relative_path = '../PickleFiles/{sentiment}_sentiment.pickle'
    relative_path = default_relative_path.format(sentiment = classifier_identifier)
    with open(relative_path, 'rb') as pickle_file:
        classifier = pickle.load(pickle_file)

    return lambda x: classifier.predict(x)

class ChatBotBuilder:
    def __init__(self):
        self.is_reset_classifiers = False
        self.embedding_function = lambda x: default_model.encode(x)
        self.response_functions = {
            "standard": lambda x: default_chatbot_response_function(x),
            "negative": lambda x: default_chatbot_response_function(x),
            "aggressive": lambda x: default_chatbot_response_function(x),
            }

    def set_embedding_function(self, embedding_function):
        self.embedding_function = embedding_function

    def set_llm_response_function(self, response_type, standard_response_function):
        self.response_function["response_type"] = standard_response_function

    def reset_classifiers(self):
        self.is_reset_classifiers = True

    def build_chatbot(self):
        negative_classifier_function = construct_classifier("negative", self.is_reset_classifiers)

        response_functions = {}
        for key, value in self.response_functions.items():
            response_functions[key] = value

        embedding_function = self.embedding_function

        return ChatBot(embedding_function, response_functions, negative_classifier_function)

chat_bot_builder = ChatBotBuilder()
chat_bot = chat_bot_builder.build_chatbot()
print(chat_bot.query("What is the tallest building in NYC?"))