"""Small random architecture tests mechanics only; never a competition model."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

from protector.constants import LABELS
from protector.training import EncodedPrompts, TrainConfig, run_training


class TrainingSmoke(unittest.TestCase):
    def test_head_tail_preserves_end_and_training_saves_real_checkpoint(self):
        temp_root = Path(__file__).resolve().parents[1] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            path = Path(td)
            source = path / "random_test_fixture"
            source.mkdir()
            vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "middle", "end"]
            (source / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
            tokenizer = DistilBertTokenizerFast(vocab_file=str(source / "vocab.txt"), do_lower_case=True)
            # DistilBERT does not accept token_type_ids.
            tokenizer.model_input_names = ["input_ids", "attention_mask"]
            tokenizer.save_pretrained(source)
            config = TrainConfig(model_id="distilbert/distilbert-base-multilingual-cased", epochs=1, batch_size=3, eval_batch_size=3, accumulation_steps=2, max_length=16, num_threads=1)
            frame = pd.DataFrame({"id": [str(i) for i in range(9)], "text": ["hello " + "middle " * 25 + "end"] * 9, "language": ["pt"] * 9, "label": LABELS * 3})
            encoded = EncodedPrompts(frame, tokenizer, config)
            self.assertEqual(encoded.truncated, 9)
            self.assertEqual(encoded[0]["input_ids"][-2], vocabulary.index("end"))
            architecture = DistilBertConfig(vocab_size=len(vocabulary), max_position_embeddings=32, n_layers=1, n_heads=2, dim=16, hidden_dim=24, num_labels=3)
            DistilBertForSequenceClassification(architecture).save_pretrained(source)
            run_training(config, source, frame.iloc[:6], frame.iloc[6:].reset_index(drop=True), path / "run", {"synthetic_fixture": True}, synthetic_test=True)
            metric = json.loads((path / "run/metrics.json").read_text())
            self.assertTrue(0 <= metric["macro_f1"] <= 1)
            self.assertTrue((path / "run/checkpoint/model.safetensors").exists())
            self.assertEqual(len(pd.read_csv(path / "run/validation_predictions.csv")), 3)


if __name__ == "__main__":
    unittest.main()
