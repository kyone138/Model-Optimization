import transformers
print(transformers.__version__)
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
import re


tokenizer = AutoTokenizer.from_pretrained("olmo_finetuned2/checkpoint-31000", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("olmo_finetuned2/checkpoint-31000", trust_remote_code=True)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer,device=0, max_new_tokens=100)
with open("test_piqa.txt") as file:
    lines = [line.rstrip() for line in file]

pred_ans = generator(lines[:3087], do_sample=False)

with open('preds_piqa.txt', 'w', encoding="UTF-8") as f:
    for pre in pred_ans:
        line = pre[0]['generated_text']
        # print(line)
        
        # need to do this for hotpot 
        # gen_answer = line.split("\tAnswer:")[-1]

        # pattern for hotpot 
        # pattern = r'^(.*?)Question:'

        # pattern for openbook and piqa
        pattern = r'Answer: (\w)'
        
        # Use re.findall to find all matches of the pattern in the text
        matches = re.findall(pattern, line)
        
        if matches:
            clean_answer = matches[0].strip()
            # print(clean_answer) 
        
        f.write(f"{clean_answer}\n")