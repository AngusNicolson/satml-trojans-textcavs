
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def main():
    class_names_long_path = "data/class_names.txt"
    with open(class_names_long_path, "r") as fp:
        class_names = fp.read()

    targets = class_names.split("\n")
    max_response_length = 200
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-dpo-13b")
    model = AutoModelForCausalLM.from_pretrained("allenai/tulu-2-dpo-13b", quantization_config=bnb_config, device_map='cuda')

    base_prompt = """<|user|>
{msg}
<|assistant|>
"""

    seen_around_prompt = """In your responses, do not include more than one thing in each line and be succinct where possible. Do not include the original object name. Only answer this question three times.
1. List the things most commonly seen around a "tench, Tinca tinca":
- a pond
- fish
- a net
- a rod
- a reel
- a hook
- bait

2. List the things most commonly seen around a "beer glass":
- beer
- a bar
- a coaster
- a napkin
- a straw
- a lime
- a person

3. List the things most commonly seen around a "{target}":"""

    seen_around_responses = get_all_responses(
        base_prompt,
        seen_around_prompt,
        targets,
        model,
        tokenizer,
        max_response_length=200,
        log=True
    )
    with open("data/text_concepts/tulu_4bit_superclass_00.json", "w") as fp:
        json.dump(seen_around_responses, fp, indent=2)

    part_of_prompt = """In your responses, do not include more than one thing in each line and be succinct where possible. Do not include the original object name. Only answer this question three times.
1. List visual elements or parts of a "tench, Tinca tinca":
- scales
- shiny
- eyes
- fins
- gills

2. List visual elements or parts of a "beer glass":
- beer
- glass
- handle
- reflective
- brown
- transparent

3. List visual elements or parts of a "{target}":"""

    superclass_prompt = """In your responses, do not include more than one thing in each line and be succinct where possible. Do not include the original object name. Only answer this question three times.
1. Give superclasses for the word "tench, Tinca tinca":
- fish
- carp
- animal
- vertebrate

2. Give superclasses for the word "beer glass":
- container
- glass
- cup
- object

3. Give superclasses for the word "{target}":"""

    similar_to_prompt = """In your responses, do not include more than one thing in each line and be succinct where possible. Do not include the original object name. Only answer this question three times.
1. Give words similar to "tench, Tinca tinca":
- fish
- carp
- goldfish
- salmon

2. Give words similar to "beer glass":
- wine glass
- alcohol
- beer pong
- cider

3. Give words similar to "{target}":
    """

    superclass_responses = get_all_responses(
        base_prompt,
        superclass_prompt,
        targets,
        model,
        tokenizer,
        max_response_length=200,
        log=True
    )
    with open("data/text_concepts/tulu_4bit_superclass_00.json", "w") as fp:
        json.dump(superclass_responses, fp, indent=2)

    part_of_responses = get_all_responses(
        base_prompt,
        part_of_prompt,
        targets,
        model,
        tokenizer,
        max_response_length=200,
        log=True
    )
    with open("data/text_concepts/tulu_4bit_part_of_00.json", "w") as fp:
        json.dump(part_of_responses, fp, indent=2)

    similar_to_responses = get_all_responses(
        base_prompt,
        similar_to_prompt,
        targets,
        model,
        tokenizer,
        max_response_length=200,
        log=True
    )
    with open("data/text_concepts/tulu_4bit_similar_to_00.json", "w") as fp:
        json.dump(similar_to_responses, fp, indent=2)

    print("Done!")


def get_all_responses(base_prompt, prompt_template, targets, model, tokenizer, max_response_length=200, log=True):
    outputs = {}
    for target in targets:
        input_text = prompt_template.replace("{target}", target)
        prompt = base_prompt.replace('{msg}', input_text)
        generated_output = get_response_for_single_prompt(
            prompt,
            model,
            tokenizer,
            max_response_length=max_response_length
        )
        outputs[target] = generated_output
        if log:
            print(generated_output)
    return outputs


def get_response_for_single_prompt(prompt, model, tokenizer, max_response_length=200):
    with torch.no_grad():
        output = model.generate(
            **tokenizer(
                prompt,
                return_tensors="pt").to("cuda"),
            max_new_tokens=max_response_length
        )
    generated_output = tokenizer.decode(output[0])
    del output
    return generated_output


if __name__ == "__main__":
    main()
