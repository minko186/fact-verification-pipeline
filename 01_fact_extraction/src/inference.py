import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pprint import pprint


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Support both old flat config and new nested config schema
    model_id = (
        config.get("model", {}).get("hub_model_id")
        or config.get("training_args", {}).get("hub_model_id")
        or config.get("model", {}).get("name")
        or config.get("model_name")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    return tokenizer, model, device


def generate_claim(evidence_text, tokenizer, model, device, config=None,
                   max_input_length=None, max_output_length=None):
    # Read generation params from config if provided, else fall back to defaults
    if config is not None:
        prompt_template = config.get("prompt", {}).get(
            "template", "extract fact: {evidence} Based on this, what is the claim?"
        )
        max_input_length = max_input_length or config.get("prompt", {}).get("max_input_length", 512)
        max_output_length = max_output_length or config.get("prompt", {}).get("max_output_length", 128)
        num_beams = config.get("inference", {}).get("num_beams", 4)
    else:
        prompt_template = "extract fact: {evidence} Based on this, what is the claim?"
        max_input_length = max_input_length or 512
        max_output_length = max_output_length or 128
        num_beams = 4

    input_text = prompt_template.format(evidence=evidence_text)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    config = load_config("config.yaml")
    tokenizer, model, device = load_model(config)
    example_evidences = [
        """Through the Looking-Glass, and What Alice Found There is a novel published in December 1871 by Lewis Carroll, the pen name of Charles Lutwidge Dodgson, a mathematics lecturer at Christ Church, Oxford. It is the sequel to his Alice's Adventures in Wonderland (1865), in which many of the characters were anthropomorphic playing cards. In this second novel, the theme is chess. As in the earlier book, the central figure, Alice, enters a fantastical world, this time by climbing through a large looking-glass (a mirror)[n 1] into a world that she can see beyond. There she finds that, just as in a reflection, things are reversed, including logic (for example, running helps one remain stationary, walking away from something brings one towards it, chessmen are alive and nursery-rhyme characters are real).""",
        """Top Chef was launched on Bravo in 2006 and featured civilians called 'cheftestants' competing for $100,000, a feature in Food & Wine magazine, and a showcase at the Food & Wine Classic in Aspen.[1] The programme frequently had guests as judges, prompting the programme's judges Tom Colicchio and Hubert Keller to consider mounting a derivative of the programme for professionals.""",
        """From 1986 to 1989, Karki worked as assistant teacher at Mahendra Multiple Campus, Dharan; from 1988, she concurrently was the bar president of the Koshi Zonal Court until 1990.[5][4] That year, she participated in the 1990 People's Movement to overthrow the Panchayat regime and was imprisoned in Biratnagar Jail.""",
    ]

    for evidence in example_evidences:
        pprint(evidence)
        claim = generate_claim(evidence, tokenizer, model, device, config=config)
        print("Generated Claim:", claim)
        print("-" * 50)
