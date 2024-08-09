from flask import Flask, request, jsonify
import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import argparse



# Parse command-line arguments
parser = argparse.ArgumentParser(description='EAGLE API Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind the server')
parser.add_argument('--port', type=int, default=7865, help='Port number to run the server on')
parser.add_argument('--base-model-path', type=str, required=True, help='Path to the base model')
parser.add_argument('--ea-model-path', type=str, required=True, help='Path to the EAGLE checkpoint')
args = parser.parse_args()

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_text():
	global ea_model

	data = request.get_json()
	messages = data.get('messages', [])
	max_new_tokens = data.get('max_new_tokens', 512)
	temperature = data.get('temperature', 0.0)
	top_p = data.get('top_p', 0.0)
	top_k = data.get('top_k', 0)

	# Create a conversation template based on the model type
	conv = get_conversation_template(ea_model.base_model_name_or_path)

	# Add messages to the conversation template
	for message in messages:
		role = message.get('role', '')
		content = message.get('content', None)
		conv.append_message(conv.roles[0] if role.lower() == "user" else conv.roles[1], content)
	conv.append_message(conv.roles[1], '')

	# Get the prompt from the conversation template
	prompt = conv.get_prompt()
	#print('prompt:', prompt)

	input_ids = ea_model.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).cuda()
	#attention_mask = torch.ones_like(input_ids)

	with torch.inference_mode():
		output_ids = ea_model.eagenerate(
			input_ids=input_ids,
			#attention_mask=attention_mask,
			is_llama3=True,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
			max_new_tokens=max_new_tokens
		)

	generated_text = ea_model.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
	#return jsonify({'generated_text': generated_text})
	return jsonify({'choices': [
		{'message': {
			'content': generated_text,
			'role': 'assistant',
		}},
	]})


if __name__ == '__main__':
	global ea_model
	# Load the base model and EAGLE checkpoint
	ea_model = EaModel.from_pretrained(
		base_model_path=args.base_model_path,
		ea_model_path=args.ea_model_path,
		torch_dtype=torch.float16,
		device_map='cuda',
	)
	ea_model.eval()

	app.run(host=args.host, port=args.port, debug=False, threaded=False)
