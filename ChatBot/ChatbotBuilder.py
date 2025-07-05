from sklearn.tree import DecisionTreeClassifier
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

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

def ChatBot():
    def __init__(self, embedding_function, response_functions: dict):
        pass

    def query(self, query):
        pass


def construct_classifier():
    pass

class ChatBotBuilder:
    def __init__(self):
        self.is_reset_classifiers = False
        self.embedding_function = lambda x: default_model.encode(x)
        self.response_functions = {
            "standard": lambda x: default_chatbot_response_function(x),
            "negative": lambda x: default_chatbot_response_function(x),
            "aggressive": lambda x: default_chatbot_response_function(x),
            }

    def set_embedding_method(self, embedding_method):
        self.embedding_method = embedding_method

    def set_llm_response_function(self, response_type, standard_response_function):
        self.response_function["response_type"] = standard_response_function

    def reset_classifiers(self):
        self.is_reset_classifiers = True

    def build_chatbot(self):
        negative_classifier_method = construct_classifier(self.embedding_method)
        pass