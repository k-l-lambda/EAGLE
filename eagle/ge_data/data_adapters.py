
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


class LmsysChatAdatper:
	loader_type = 'parquet'

	@staticmethod
	def iterate (examples):
		for conversation in examples['conversation']:
			while conversation[0]['role'] != 'user':
				conversation = conversation[1:]

			yield conversation


class UltraChatAdatper:
	loader_type = None

	@staticmethod
	def iterate (examples):
		for conversation in examples['data']:
			messages = []
			for i, sentence in enumerate(conversation):
				role = 'user' if i % 2 == 0 else 'assistant'
				messages.append(
					dict(
						role=role,
						content=sentence,
					)
				)

			yield messages


adapter_dict = dict(
	sharegpt=ShareGPTAdatper,
	lmsyschat=LmsysChatAdatper,
	ultrachat=UltraChatAdatper,
)
