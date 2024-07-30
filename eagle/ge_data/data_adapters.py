
class ShareGPTAdatper:
	loader_type = 'json'

	@staticmethod
	def iterate (examples):
		roles = {"human": "user", "gpt": "assistant"}

		for i in range(len(examples['id'])):
			source = examples['conversations'][i]

			while source[0]['from'] != 'human':
				source = source[1:]

			messages = list(map(lambda sentence: dict(
				role=roles[sentence['from']],
				content=sentence['value']
			), source))

			yield messages
