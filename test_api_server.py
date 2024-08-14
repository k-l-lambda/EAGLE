import unittest
import requests
import asyncio
import tqdm.asyncio
import httpx
import time


client = httpx.AsyncClient(timeout=None)

async def request(sem, url, payload):
	async with sem:
		response = await client.post(url, json=payload)
		data = response.json()
		#print('generated_text:', data["choices"][0])
		#time.sleep(1)

		#print(f'{data=}')
		#return data['choices'][0]['message']['content']
		return len(data['choices'][0]['message']['content'])


class TestAPIServer(unittest.TestCase):
	def setUp(self):
		self.api_url = "http://localhost:7865/generate"

	def _test_generate_text(self):
		payload = {
			"messages": [
				{"role": "user", "content": "Who is Larry Page?"},
			],
			"max_new_tokens": 512,
			"temperature": 0,
			#"top_p": 0.9,
			#"top_k": 50
		}

		response = requests.post(self.api_url, json=payload)
		data = response.json()
		print('generated_text:', data["choices"][0])

	def test_generate_text_concurrent(self):
		async def process ():
			payload = {
				"messages": [
					{"role": "user", "content": "Who is Larry Page?"},
				],
				"max_new_tokens": 512,
				"temperature": 1,
			}

			sem = asyncio.Semaphore(3)
			futures = []
			for i in range(20):
				futures.append(request(sem, self.api_url, payload))

			batch = await tqdm.asyncio.tqdm.gather(*futures, leave=False)
			print('batch:', batch)

		asyncio.run(process())

if __name__ == "__main__":
	unittest.main()
