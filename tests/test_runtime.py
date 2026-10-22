import os
import unittest
from unittest.mock import patch

from protector.runtime import ENTRY_MODEL_ID, require_kaggle_runtime


class RuntimeRulesTests(unittest.TestCase):
    def test_local_runtime_is_blocked_even_with_an_approved_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALL competition training"):
                require_kaggle_runtime(ENTRY_MODEL_ID)

    def test_kaggle_entry_rejects_an_additional_backbone(self):
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}), patch("protector.runtime.platform.system", return_value="Linux"), patch("protector.runtime.Path.is_dir", return_value=True):
            self.assertEqual(require_kaggle_runtime(ENTRY_MODEL_ID)["platform"], "kaggle")
            with self.assertRaisesRegex(ValueError, "exactly one backbone"):
                require_kaggle_runtime("distilbert/distilbert-base-multilingual-cased")


if __name__ == "__main__":
    unittest.main()
