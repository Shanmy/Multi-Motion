from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)


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

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}


# Inference is also possible via Transformers' pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)

pipe_output = pipe(prompt_template)[0]['generated_text']
print("pipeline output: ", pipe_output)
