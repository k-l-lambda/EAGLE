import torch
import torch.multiprocessing as mp
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
from flask import Flask, request, jsonify
import argparse



# Parse command-line arguments
parser = argparse.ArgumentParser(description='EAGLE API Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind the server')
parser.add_argument('--port', type=int, default=7865, help='Port number to run the server on')
parser.add_argument('--base-model-path', type=str, required=True, help='Path to the base model')
parser.add_argument('--ea-model-path', type=str, required=True, help='Path to the EAGLE checkpoint')
args = parser.parse_args()


app = Flask(__name__)


def worker (device_index, queue):
	# Load EAGLE model on the specific device
	device = f'cuda:{device_index}'

	model = EaModel.from_pretrained(
		base_model_path=args.base_model_path,
		ea_model_path=args.ea_model_path,
		torch_dtype=torch.float16,
		device_map=device,
	)
	model.eval()

	while True:
		input = queue.get()
		if input is None:
			break

		print('got:', device_index)

		# Create a conversation template based on the model type
		conv = get_conversation_template(model.base_model_name_or_path)

		# Add messages to the conversation template
		for message in input['messages']:
			role = message.get('role', '')
			content = message.get('content', None)
			conv.append_message(role, content)
		conv.append_message(conv.roles[1], '')

		# Get the prompt from the conversation template
		prompt = conv.get_prompt()

		input_ids = model.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)

		with torch.inference_mode():
			output_ids = model.eagenerate(
				input_ids=input_ids,
				is_llama3=True,
				temperature=input['temperature'],
				top_p=input['top_p'],
				top_k=input['top_k'],
				max_new_tokens=input['max_new_tokens'],
			)

		generated_text = model.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

		queue.put(generated_text)


@app.route('/generate', methods=['POST'])
def generate ():
	global request_i, num_gpus

	data = request.get_json()
	messages = data.get('messages', [])
	max_new_tokens = data.get('max_new_tokens', 512)
	temperature = data.get('temperature', 0.0)
	top_p = data.get('top_p', 0.0)
	top_k = data.get('top_k', 0)

	input = dict(
		messages=messages,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
	)

	queue = device_queues[request_i % num_gpus]
	request_i += 1
	queue.put(input)
	output = queue.get()

	return jsonify({'choices': [
		{'message': {
			'content': output,
			'role': 'assistant',
		}},
	]})


if __name__ == '__main__':
	global request_i, num_gpus

	num_gpus = torch.cuda.device_count()
	request_i = 0
	device_queues = {}
	processes = []

	mp.set_start_method('spawn', force=True)

	for i in range(num_gpus):
		queue = mp.Queue()
		device_queues[i] = queue
		p = mp.Process(target=worker, args=(i, queue))
		p.start()
		processes.append(p)

	app.run(host=args.host, port=args.port, debug=False, threaded=True)

	# Cleanup
	for i in range(num_gpus):
		device_queues[i].put(None)
	for p in processes:
		p.join()
