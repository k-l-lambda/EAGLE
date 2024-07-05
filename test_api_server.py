import unittest
import json
import requests

class TestAPIServer(unittest.TestCase):
	def setUp(self):
		self.api_url = "http://localhost:7865/generate"

	def test_generate_text(self):
		payload = {
			"messages": [
				{"role": "human", "content": "Can you tell me about the EAGLE model?"}
			],
			"max_new_tokens": 100,
			"temperature": 0.7,
			"top_p": 0.9,
			"top_k": 50
		}

		response = requests.post(self.api_url, json=payload)
		data = response.json()
		print(data["generated_text"])

if __name__ == "__main__":
	unittest.main()
