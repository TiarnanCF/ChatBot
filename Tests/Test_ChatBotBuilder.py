import unittest
from ChatBot.ChatBotBuilder import ChatBot, ChatBotBuilder
import numpy as np

class TestChatBotBuilder(unittest.TestCase):
  tolerance = 0.000000001

  def setUp(self):
    self.chatbot_builder = ChatBotBuilder()

  def test_build_basic(self):
    chatbot = self.chatbot_builder.build_chatbot()
    self.assertTrue(isinstance(chatbot,ChatBot))

if __name__ == '__main__':
  unittest.main()