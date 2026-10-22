import ast
import base64
import json
from pathlib import Path
import unittest
import zlib

import nbformat

from protector.constants import APPROVED, ENCODERS


class NotebookArtifactTests(unittest.TestCase):
    def test_selected_backbone_and_embedded_source_match_repository(self):
        root = Path(__file__).resolve().parents[1]
        team = json.loads((root / "configs/team.json").read_text(encoding="utf-8"))
        notebook = nbformat.read(root / "notebooks" / (team["notebook_title"] + ".ipynb"), as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(notebook.metadata.title, team["notebook_title"])
        literals = {}
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            compile(cell.source, "<notebook cell>", "exec")
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"MODEL_ID", "CONFIG_PATHS", "payload"}:
                            literals[target.id] = ast.literal_eval(node.value)
        model_id = literals["MODEL_ID"]
        self.assertIn(model_id, ENCODERS)
        self.assertIn(f"`{model_id}`", notebook.cells[0].source)
        self.assertIn("https://www.kaggle.com/models/" + APPROVED[model_id], notebook.cells[0].source)
        embedded = json.loads(zlib.decompress(base64.b64decode(literals["payload"], validate=True)))
        expected = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in (root / "protector").glob("*.py")}
        for relative in set(literals["CONFIG_PATHS"]) | {"configs/team.json"}:
            expected[relative] = (root / relative).read_text(encoding="utf-8")
        self.assertEqual(embedded, expected, "Notebook must embed current source/config only, with no competition row/split payload")
        for config_path in literals["CONFIG_PATHS"]:
            self.assertEqual(json.loads(embedded[config_path])["model_id"], model_id)
        self.assertNotIn("configs/distilbert_ce.json", embedded)


if __name__ == "__main__":
    unittest.main()
