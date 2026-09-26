"""Candidate-aware lexical features; deliberately avoids corpus-wide matrices."""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, distance


STOP_TOKENS = {
    "the", "inc", "llc", "ltd", "limited", "corp", "corporation", "group", "company", "co",
    "global", "international", "services", "holdings", "enterprises", "associates", "solutions",
    "and", "pte", "pvt", "private", "gmbh", "sa", "bv", "sl", "ag", "technologies",
    "street", "road", "rd", "st", "avenue", "ave", "lane", "drive", "dr", "near", "opposite",
}


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def _pair_features(left: str, right: str, prefix: str) -> dict[str, float]:
    a, b = _tokens(left), _tokens(right)
    sa, sb = set(a), set(b)
    union = sa | sb
    informative_a, informative_b = sa - STOP_TOKENS, sb - STOP_TOKENS
    informative_union = informative_a | informative_b
    first_informative_a = next((token for token in a if token not in STOP_TOKENS), "")
    first_informative_b = next((token for token in b if token not in STOP_TOKENS), "")
    digits_a = "".join(c for c in left if c.isdigit())
    digits_b = "".join(c for c in right if c.isdigit())
    number_a = re.search(r"\b\d{1,8}\b", left)
    number_b = re.search(r"\b\d{1,8}\b", right)
    postal_a = re.findall(r"\b\d{5,10}\b", left)
    postal_b = re.findall(r"\b\d{5,10}\b", right)
    initials_a = "".join(word[0] for word in a if word)
    initials_b = "".join(word[0] for word in b if word)
    return {
        f"{prefix}_ratio": fuzz.ratio(left, right) / 100,
        f"{prefix}_partial_ratio": fuzz.partial_ratio(left, right) / 100,
        f"{prefix}_token_sort_ratio": fuzz.token_sort_ratio(left, right) / 100,
        f"{prefix}_token_set_ratio": fuzz.token_set_ratio(left, right) / 100,
        f"{prefix}_jaro_winkler": distance.JaroWinkler.similarity(left, right),
        f"{prefix}_jaccard": len(sa & sb) / max(1, len(union)),
        f"{prefix}_shared_tokens": float(len(sa & sb)),
        f"{prefix}_informative_jaccard": len(informative_a & informative_b) / max(1, len(informative_union)),
        f"{prefix}_informative_shared_tokens": float(len(informative_a & informative_b)),
        f"{prefix}_containment": len(sa & sb) / max(1, min(len(sa), len(sb))),
        f"{prefix}_length_delta": float(abs(len(left) - len(right))),
        f"{prefix}_normalized_length_delta": abs(len(left) - len(right)) / max(1, max(len(left), len(right))),
        f"{prefix}_token_count_delta": float(abs(len(a) - len(b))),
        f"{prefix}_exact": float(bool(left) and left == right),
        f"{prefix}_first_token_equal": float(bool(a and b and a[0] == b[0])),
        f"{prefix}_first_informative_token_equal": float(bool(first_informative_a and first_informative_b)
                                                            and first_informative_a == first_informative_b),
        f"{prefix}_last_token_equal": float(bool(a and b and a[-1] == b[-1])),
        f"{prefix}_digit_equal": float(digits_a == digits_b),
        f"{prefix}_digit_jaccard": len(set(digits_a) & set(digits_b)) / max(1, len(set(digits_a) | set(digits_b))),
        f"{prefix}_prefix4_equal": float(bool(left and right) and left[:4] == right[:4]),
        f"{prefix}_initials_equal": float(bool(initials_a and initials_b) and initials_a == initials_b),
        f"{prefix}_house_number_equal": float(bool(number_a and number_b) and number_a.group() == number_b.group()),
        f"{prefix}_postal_equal": float(bool(set(postal_a) & set(postal_b))),
    }


class FeatureExtractor:
    """Extract features only for retrieved pairs, in bounded chunks."""

    def __init__(self, chunk_size: int = 50_000):
        if chunk_size < 1:
            raise ValueError("feature chunk size must be positive")
        self.chunk_size = chunk_size

    @staticmethod
    def feature_columns() -> list[str]:
        return [f"{field}_{feature}" for field in ("name", "addr") for feature in (
            "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
            "jaro_winkler", "jaccard", "shared_tokens", "informative_jaccard",
            "informative_shared_tokens", "containment",
            "length_delta", "normalized_length_delta", "token_count_delta", "exact",
            "first_token_equal", "first_informative_token_equal", "last_token_equal", "digit_equal", "digit_jaccard",
            "prefix4_equal", "initials_equal", "house_number_equal", "postal_equal")
        ] + ["same_country", "exact_name_country", "exact_address_country",
             "name_addr_agreement", "name_quality", "address_quality",
             "missing_name", "missing_address", "target_source", "block_count",
             "block_exact_name", "block_name_token", "block_name_prefix",
             "block_address_number", "block_exact_address", "block_address_token"]

    def iter_candidate_features(self, candidate_df: pd.DataFrame,
                                df_ref: pd.DataFrame, df_target: pd.DataFrame):
        """Yield numeric feature frames without retaining prior chunks.

        Candidate batches and the corresponding entity lookup frames should be
        supplied at the configured chunk size by disk-backed callers.
        """
        required = {"source1_entity_id", "target_entity_id"}
        if not required.issubset(candidate_df.columns):
            raise ValueError(f"Candidates must contain {sorted(required)}")
        entity_columns = {"entity_id", "clean_name", "clean_address", "clean_country"}
        for label, frame in (("reference", df_ref), ("target", df_target)):
            missing = entity_columns - set(frame.columns)
            if missing:
                raise ValueError(f"{label} entities are missing fields: {sorted(missing)}")
        ref = df_ref.set_index("entity_id", drop=False)
        target = df_target.set_index("entity_id", drop=False)
        for start in range(0, len(candidate_df), self.chunk_size):
            pairs = candidate_df.iloc[start:start + self.chunk_size]
            left = pairs["source1_entity_id"].map(ref["clean_name"])
            right = pairs["target_entity_id"].map(target["clean_name"])
            addr_l = pairs["source1_entity_id"].map(ref["clean_address"])
            addr_r = pairs["target_entity_id"].map(target["clean_address"])
            country_l = pairs["source1_entity_id"].map(ref["clean_country"])
            country_r = pairs["target_entity_id"].map(target["clean_country"])
            result = []
            flags = pairs.get("block_flags", pd.Series(0, index=pairs.index)).fillna(0).astype(int)
            for n, a, b, c, e, f, target_id, flag in zip(
                left.fillna(""), right.fillna(""), addr_l.fillna(""), addr_r.fillna(""),
                country_l.fillna(""), country_r.fillna(""), pairs["target_entity_id"], flags, strict=True
            ):
                row = _pair_features(n, a, "name")
                row.update(_pair_features(b, c, "addr"))
                row["same_country"] = float(bool(e) and e.casefold() == f.casefold())
                row["exact_name_country"] = row["name_exact"] * row["same_country"]
                row["exact_address_country"] = row["addr_exact"] * row["same_country"]
                row["name_addr_agreement"] = row["name_ratio"] * row["addr_ratio"]
                row["name_quality"] = min(1.0, len(_tokens(n)) / 4.0) * min(1.0, len(n) / 12.0)
                row["address_quality"] = min(1.0, len(_tokens(b)) / 6.0) * min(1.0, len(b) / 24.0)
                row["missing_name"] = float(not n or not a)
                row["missing_address"] = float(not b or not c)
                row["target_source"] = 2.0 if str(target_id).startswith("S2-") else 3.0
                row["block_count"] = float(int(flag).bit_count())
                for name, bit in (("exact_name", 1), ("name_token", 2), ("name_prefix", 4),
                                  ("address_number", 8), ("exact_address", 16), ("address_token", 32)):
                    row[f"block_{name}"] = float(bool(int(flag) & bit))
                result.append(row)
            block = pd.DataFrame(result, columns=self.feature_columns())
            block.insert(0, "target_entity_id", pairs["target_entity_id"].to_numpy())
            block.insert(0, "source1_entity_id", pairs["source1_entity_id"].to_numpy())
            yield block

    def generate_candidate_features(self, candidate_df: pd.DataFrame,
                                    df_ref: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        """Compatibility helper for a single bounded batch.

        Large callers must consume ``iter_candidate_features`` and persist each
        result. Refusing multi-chunk aggregation here prevents an accidental
        full-corpus feature DataFrame.
        """
        if len(candidate_df) > self.chunk_size:
            raise ValueError(
                f"Received {len(candidate_df):,} candidates, above the configured "
                f"chunk size {self.chunk_size:,}; use iter_candidate_features()"
            )
        chunks = list(self.iter_candidate_features(candidate_df, df_ref, df_target))
        if not chunks:
            return pd.DataFrame(columns=["source1_entity_id", "target_entity_id", *self.feature_columns()])
        return chunks[0]
