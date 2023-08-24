#%%
import transformers as t
import torch
import peft
import time
import wandb
#%%
tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
tokenizer.pad_token_id = 0
#%%
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = peft.get_peft_model(model, config).to(device)
# peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-600/adapter_model.bin"))
#%%

TEMPLATE = "Below is a title for a satirical article. Write an article that appropriately suits the title: \n\n### Title:\n{Report: Iran Less Than 10 Years Away From 2016}\n\n### Article:\n WASHINGTON, DC—According to an alarming new Department of Defense report combining civilian, military, and calendric evidence, Iran may be as few as nine years away from the year 2016. \"Every day they get one day closer,\" Defense Secretary Robert Gates said during a White House press conference Tuesday. \"At the rate they\'re going, they will reach 2016 at the same time as the United States—and given their geographic position relative to the international date line, possibly even sooner.\" The report recommended that the U.S. engage in bellicose international posturing, careless brinksmanship, and an eventual overwhelming series of nuclear strikes in order to prevent Iran from reaching this milestone. Below is a title for a satirical article. Write an article that appropriately suits the title: \n\n### Title:\n{Dzhokar Tsarnaev Finally Moves Off Campus}\n\n### Article:\n BOSTON—After living in residence halls during his first three semesters at the University of Massachusetts Dartmouth, sophomore student Dzhokar Tsarnaev was finally able to get a place of his own and move off campus this week, the 19-year-old told reporters. “Last semester I shared a double room with a guy at Pine Dale Hall, but now I’ve got a place off campus with no roommate, which is nice,” the engineering student said of his new living arrangements, a 10-by-10-foot room located on the first floor of a decommissioned military base about an hour and 40 minutes north from the university. “It’s been pretty sweet so far. The building is really safe, I don’t have to share a sink with anyone, and living off campus is a lot cheaper than the dorms. Of course, the downside is that the neighbors suck. But I’ve been thinking about this for a long time now, so I’m glad I finally did it.” Tsarnaev added that although he’s no longer required to be on a meal plan, he decided to sign up for the 10-meal-per week option and add some extra Dining Dollars."
INSTRUCTION = "Article About Return Of Burger King Chicken Fries Only News Area Man Has Clicked On Today"
prompt = TEMPLATE.format(instruction=INSTRUCTION)

title = str(input("title: "))

# %% 
wandb.init(
    project="Llama",
    name = title,
    config={
        "TEMPLATE": TEMPLATE,
        "INSTRUCTION": INSTRUCTION,
        "prompt": prompt,
    })
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))
wandb.log({"output": pipe(prompt)})
wandb.finish()