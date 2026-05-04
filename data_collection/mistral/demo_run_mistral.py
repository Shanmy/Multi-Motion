import mistral_utils as mu

model, tokenizer = mu.init_mistral()

description="Rick Ballou On Twitter Jaguars First Round Draft Pick Josh Allen And His Wife Kaitlyn"

val = mu.run_mistral_isGoodPoseDescription(model, tokenizer, description)

print(' ----- sentence to be evaluated -----')
print(description)
print(' ----- evaluation result ------')
print(val)

