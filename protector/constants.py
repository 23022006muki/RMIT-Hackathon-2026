LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
LANGUAGES = ["pt", "fr", "it", "en"]

# Only these pretrained backbones are permitted by the supplied 2026 Overview.
APPROVED = {
    "distilbert/distilbert-base-multilingual-cased": "aisuko/distilbert-base-multilingual-cased/transformers/default",
    "jhu-clsp/mmBERT-small": "aisuko/mmbert-small/transformers/default",
    "jhu-clsp/mmBERT-base": "aisuko/mmbert-base/transformers/default",
    "FacebookAI/xlm-roberta-base": "aisuko/xlm-roberta-base/transformers/default",
    "FacebookAI/xlm-roberta-large": "aisuko/xlm-roberta-large/transformers/default",
    "google/rembert": "aisuko/rembert/transformers/default",
    "protectai/deberta-v3-base-prompt-injection-v2": "aisuko/deberta-v3-base-prompt-injection-v2/transformers/default",
    "patronus-studio/wolf-defender-prompt-injection": "aisuko/wolf-defender-prompt-injection/transformers/default",
    "ibm-granite/granite-3.3-2b-instruct": "aisuko/granite-3.3-2b-instruct/transformers/default",
    "mistralai/Mistral-7B-Instruct-v0.3": "aisuko/mistral-7b-instruct-v0.3/transformers/default",
}

ENCODERS = set(APPROVED) - {
    "ibm-granite/granite-3.3-2b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
}
