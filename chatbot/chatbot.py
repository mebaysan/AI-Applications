"""
Transformers and LLMs work together within a chatbot to enable conversation. Here's a simplified explanation of how they interact:

Input processing: When you send a message to the chatbot, the transformer helps process your input. It breaks down your message into smaller parts and represents them in a way that the chatbot can understand. Each part is called a token.

Understanding context: The transformer passes these tokens to the LLM, which is a language model trained on lots of text data. The LLM has learned patterns and meanings from this data, so it tries to understand the context of your message based on what it has learned.

Generating response: Once the LLM understands your message, it generates a response based on its understanding. The transformer then takes this response and converts it into a format that can be easily sent back to you.

Iterative conversation: As the conversation continues, this process repeats. The transformer and LLM work together to process each new input message, understand the context, and generate a relevant response.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# model is an instance of the class AutoModelForSeq2SeqLM, 
# which allows you to interact with your chosen language model.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# tokenizer is an instance of the class AutoTokenizer, 
# which optimizes your input and passes it to the language model efficiently. 
# It does so by converting your text input to “tokens”, which is how the model interprets the text.
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

history_string = "".join(conversation_history)

input_text = "Hello, how are you doing?"

inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

#tokenizer.pretrained_vocab_files_map

outputs = model.generate(**inputs)
print(outputs)

response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)