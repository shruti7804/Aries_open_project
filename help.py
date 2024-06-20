import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

model_dir = "pretrained/model"
tokenizer_dir = "pretrained/tokenizer"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.save_pretrained(model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(tokenizer_dir)

