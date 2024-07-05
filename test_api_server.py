import unittest
import json
import requests

class TestAPIServer(unittest.TestCase):
	def setUp(self):
		self.api_url = "http://localhost:7865/generate"

	def test_generate_text(self):
		payload = {
			"messages": [
				{"role": "human", "content": "What are the names of some famous actors that started their careers on Broadway?"},
				{"role": "assistant"},
			],
			"max_new_tokens": 512,
			"temperature": 0,
			#"top_p": 0.9,
			#"top_k": 50
		}

		response = requests.post(self.api_url, json=payload)
		data = response.json()
		print('generated_text:', data["generated_text"])

if __name__ == "__main__":
	unittest.main()
