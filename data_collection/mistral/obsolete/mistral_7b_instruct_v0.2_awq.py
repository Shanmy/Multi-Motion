from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

# Using the text streamer to stream output one token at a time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "system: please decide whether the user's description is desriptive of human pose. Please answer in yes or no, without any explanation. user:  "
prompt = """system: Given a user's sentence, please describe whether the sentence is descriptive of human pose. Please answer using just yes or no. Following are some Demonstrations.
Demonstrations: 
Q: Intelligentsia Baristas.
A: no

Q: Many attendees also returned to view the 2015 SAM Members Show in the Gallery space.
A: no

Q: two men running in a race together at a race
A: yes

Q: a group of people are playing basketball
A: yes
"""

#prompt+="user: there is a person or group of people standing in front of a row of ancient Chinese statues. They appear to be looking at the statues, possibly admiring their craftsmanship and historical significance. The people are positioned in various body postures, with some standing upright and others leaning towards the statues for a closer look."
prompt+="user:Rick Ballou On Twitter Jaguars First Round Draft Pick Josh Allen And His Wife Kaitlyn"
prompt_template=f'''<s>[INST] {prompt} [/INST]
'''

# Convert prompt to tokens
tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}

# Generate streamed output, visible one token at a time
generation_output = model.generate(
    tokens,
    streamer=streamer,
    **generation_params
)

# Generation without a streamer, which will include the prompt in the output
generation_output = model.generate(
    tokens,
    **generation_params
)

# Get the tokens from the output, decode them, print them
token_output = generation_output[0]
text_output = tokenizer.decode(token_output)
print("model.generate output: ", text_output)

# Inference is also possible via Transformers' pipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)

pipe_output = pipe(prompt_template)[0]['generated_text']
print("pipeline output: ", pipe_output)
