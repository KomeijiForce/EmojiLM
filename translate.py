from transformers import BartTokenizer, BartForConditionalGeneration

def translate(sentence, **argv):
    inputs = tokenizer(sentence, return_tensors="pt")
    generated_ids = generator.generate(inputs["input_ids"], **argv)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace(" ", "")
    return decoded

path = "KomeijiForce/bart-large-emojilm"
tokenizer = BartTokenizer.from_pretrained(path)
generator = BartForConditionalGeneration.from_pretrained(path)

sentence = "I love the weather in Alaska!"
decoded = translate(sentence, num_beams=4, do_sample=True, max_length=100)
print(decoded)
