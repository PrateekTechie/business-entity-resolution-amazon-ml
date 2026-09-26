import unittest
import pandas as pd

from src.preprocessing.text_cleaner import TextCleaner


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_names_addresses_and_open_set_country(self):
        frame = pd.DataFrame({
            "entity_id": ["S1-a"], "business_name": ["Acme, Inc."],
            "business_address": ["12 Main St."], "country": ["France"],
        })
        actual = TextCleaner().preprocess_dataset(frame).iloc[0]
        self.assertEqual(actual.clean_name, "acme incorporated")
        self.assertEqual(actual.clean_address, "12 main street")
        self.assertEqual(actual.clean_country, "france")


if __name__ == "__main__":
    unittest.main()
