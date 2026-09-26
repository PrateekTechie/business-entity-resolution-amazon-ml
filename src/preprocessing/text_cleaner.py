import json
import os
import re
import string
import pandas as pd


class TextCleaner:

  def __init__(self, mapping_filepath: str = "config/text_mappings.json", verbose: bool = False):
    self.business_map = {}
    self.address_map = {}
    self.verbose = verbose

    if (
        os.path.exists(mapping_filepath)
        and os.path.getsize(mapping_filepath) > 0
    ):
      try:
        with open(mapping_filepath, "r") as f:
          mappings = json.load(f)
          self.business_map = mappings.get("business_suffixes", {})
          self.address_map = mappings.get("address_terms", {})
      except Exception as e:
        raise ValueError(f"Failed to read text mappings from {mapping_filepath}: {e}") from e

  def _vectorized_clean(
      self, series: pd.Series, mapping: dict = None
  ) -> pd.Series:
    """Vectorized string cleaning optimized for millions of rows in Pandas."""
    # Convert to string, lowercase, and handle NaN
    s = series.fillna("").astype(str).str.lower().str.strip()

    # Strip punctuation via regex pattern
    s = s.str.replace(f"[{re.escape(string.punctuation)}]", " ", regex=True)

    # Apply word expansions if mapping is provided
    if mapping:
      # Build a single regex replacement pattern: \b(pvt|ltd|inc)\b
      sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
      pattern = r"\b(" + "|".join(map(re.escape, sorted_keys)) + r")\b"
      s = s.str.replace(
          pattern, lambda m: mapping.get(m.group(0), m.group(0)), regex=True
      )

    # Collapse multiple spaces into a single space
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

  def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    """Cleans TSV dataframe using fast vectorized operations."""
    df_clean = df.copy()

    # Clean Business Name
    if self.verbose:
      print("  -> Cleaning business names...")
    df_clean["clean_name"] = self._vectorized_clean(
        df_clean["business_name"], self.business_map
    )

    # Clean Business Address
    if self.verbose:
      print("  -> Cleaning business addresses...")
    df_clean["clean_address"] = self._vectorized_clean(
        df_clean["business_address"], self.address_map
    )

    # Normalize Country
    if "country" in df_clean.columns:
      df_clean["clean_country"] = (
          df_clean["country"].fillna("").astype(str).str.lower().str.strip()
      )

    # Extract first word of business name for quick blocking
    df_clean["name_first_word"] = (
        df_clean["clean_name"].str.split().str[0].fillna("")
    )

    return df_clean


if __name__ == "__main__":
  cleaner = TextCleaner()
  sample = pd.DataFrame({
      "business_name": ["Xenthives AI Tech Pvt. Ltd."],
      "business_address": ["123 St. Marks Rd"],
      "country": ["IN"],
  })
  print(cleaner.preprocess_dataset(sample))
