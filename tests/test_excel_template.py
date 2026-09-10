import unittest
from pathlib import Path

from sample_data import validate_table
from volume_ui import _read_excel_template


class ExcelTemplateTests(unittest.TestCase):
    def test_documented_workbook_imports_and_preserves_volume_examples(self):
        template = Path("assets/ValidReuse_Datenvorlage.xlsx")
        self.assertTrue(template.is_file())
        raw_in, raw_out = _read_excel_template(template.read_bytes())

        inflow = validate_table(raw_in, "in")
        outflow = validate_table(raw_out, "out")

        self.assertEqual(inflow.sample_id.tolist(), ["IN-001", "IN-002"])
        self.assertEqual(outflow.value.tolist(), [3.0, 0.0, 2.0])
        self.assertEqual(outflow.effective_volume_ml.tolist(), [1875.0, 400.0, 1200.0])


if __name__ == "__main__":
    unittest.main()
