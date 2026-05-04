from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def init_mistral(): 

    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )

    return model, tokenizer

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}

def run_mistral(model, tokenizer, prompt, generation_params=generation_params, noInputEcho=True): 
    # fill prompt into prompt template
    prompt_template=f'''<s>[INST] {prompt} [/INST]
    '''

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )

    pipe_output = pipe(prompt_template)[0]['generated_text']
    #print("pipeline output: ", pipe_output)
    if noInputEcho == True: 
        val = pipe_output.removeprefix(prompt_template)
    else:
        val = pipe_output
    return val

def run_mistral_isGoodPoseDescription(model, tokenizer, description, generation_params=generation_params): 

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

    prompt += f"user: {description}"

    return run_mistral(model, tokenizer, prompt, generation_params=generation_params)

