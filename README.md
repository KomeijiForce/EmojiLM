# Fast Access

**Models:** Our models can be loaded from [bart-base-emojilm](https://huggingface.co/KomeijiForce/bart-base-emojilm) and [bart-large-emojilm](https://huggingface.co/KomeijiForce/bart-large-emojilm).

**Dataset:** Our dataset is accessible at [Text2Emoji](https://huggingface.co/datasets/KomeijiForce/Text2Emoji).

# EmojiLM
Official Implementation for "EmojiLM: Modeling the New Emoji Language"

This is a repo for models pre-trained on the [Text2Emoji](https://huggingface.co/datasets/KomeijiForce/Text2Emoji) dataset to translate setences into series of emojis.

For instance, "I love pizza" will be translated into "🍕😍".

An example implementation for translation:

```python
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
```

You will probably get some output like "❄️🏔️😍".

If you find this model & dataset resource useful, please consider cite our paper:

```
@article{DBLP:journals/corr/abs-2311-01751,
  author       = {Letian Peng and
                  Zilong Wang and
                  Hang Liu and
                  Zihan Wang and
                  Jingbo Shang},
  title        = {EmojiLM: Modeling the New Emoji Language},
  journal      = {CoRR},
  volume       = {abs/2311.01751},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2311.01751},
  doi          = {10.48550/ARXIV.2311.01751},
  eprinttype    = {arXiv},
  eprint       = {2311.01751},
  timestamp    = {Tue, 07 Nov 2023 18:17:14 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2311-01751.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
