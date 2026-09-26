import unittest
import pandas as pd

from src.blocking.attribute_blocking import AttributeBlocker
from src.preprocessing.text_cleaner import TextCleaner


class BlockingTests(unittest.TestCase):
    def test_union_is_deduplicated_and_country_is_open_set(self):
        ref = pd.DataFrame({"entity_id": ["S1-a"], "business_name": ["Acme Bakery"],
                            "business_address": ["123 Main Road"], "country": ["France"]})
        target = pd.DataFrame({"entity_id": ["S2-a"], "business_name": ["Acme Bakery"],
                               "business_address": ["123 Main Road"], "country": ["France"]})
        cleaner = TextCleaner()
        candidates = AttributeBlocker(max_group_size=20).generate_candidate_pairs(
            cleaner.preprocess_dataset(ref), cleaner.preprocess_dataset(target))
        self.assertEqual(candidates.to_records(index=False).tolist(), [("S1-a", "S2-a")])

    def test_skips_oversized_block(self):
        ref = pd.DataFrame({"entity_id": ["S1-a"], "clean_name": ["common shop"],
                            "clean_address": [""], "clean_country": ["us"]})
        target = pd.DataFrame({"entity_id": ["S2-a", "S2-b"], "clean_name": ["common shop"] * 2,
                               "clean_address": [""] * 2, "clean_country": ["us"] * 2})
        out = AttributeBlocker(max_group_size=1).generate_candidate_pairs(ref, target)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
