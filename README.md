# satml-trojans-textcavs
Entry to SaTML trojan competition 2024. 


## The explanations
Ten words for each target class are in [output/explanations/text_cav_explanations_tulu_4bit_00_cleaned.csv](output/explanations/text_cav_explanations_tulu_4bit_00_cleaned.csv). These words should be associated with the trojan. For evaluation, we believe it is easier to show just the top 5 words, as 10 can be overly confusing.

## Running the code (short)
To simply generate the explanations with pre-trained versions of the method run these scipts in order:

```bash
python generate_explanations.py
python generate_explanations.py --trojan
python compare_text_cav_results.py
```

## Running the code (long)
To run the full process, including creation of the concept list and feature aligner/converter, run these scripts in order. You will need to have downloaded ImageNet into [data/imagenet](data/imagenet).

### Concept list

```bash
python llm.py
python process_llm_concepts.py
```

### Train feature converter

```bash
python train_feature_converter.py
```

### Generate explanations
```bash
python generate_explanations.py
python generate_explanations.py --trojan
python compare_text_cav_results.py
```

## Logits

As an aside, we also checked the change in logits for a normal ResNet50 vs the ResNet50 with trojans. This proved useful for finding natural trojans.

These can be found in [output/explanations/logit_changing_images](output/explanations/logit_changing_images).

```bash
python get_imagenet_predictions.py
python compare_predictions.py
```