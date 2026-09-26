"""Bounded multi-key blocking. Candidate caps are disabled by default."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


class AttributeBlocker:
    def __init__(self, max_candidates_per_entity: int | None = None, max_group_size: int = 300,
                 max_pairs_per_block: int = 2_000_000):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_group_size = max_group_size
        self.max_pairs_per_block = max_pairs_per_block
        self.stopwords = {"THE", "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION",
                          "GROUP", "COMPANY", "CO", "GLOBAL", "INTERNATIONAL", "SERVICES",
                          "HOLDINGS", "ENTERPRISES", "ASSOCIATES", "SOLUTIONS", "AND", "PTE",
                          "PVT", "PRIVATE", "GMBH", "SA", "BV", "SL", "AG", "TECHNOLOGIES"}

    def _keys(self, df: pd.DataFrame, kind: str) -> pd.DataFrame:
        country = df.clean_country.fillna("").astype(str).str.casefold()
        name = df.clean_name.fillna("").astype(str).str.casefold()
        address = df.clean_address.fillna("").astype(str).str.casefold()
        tokens = name.str.findall(r"[\w]+")
        records = []
        for pos, (ctry, n, addr, words) in enumerate(zip(country, name, address, tokens, strict=True)):
            if kind == "exact_name" and len(n) >= 4:
                records.append((pos, f"{ctry}|{n}"))
            elif kind == "name_token":
                informative = sorted({w for w in words if len(w) >= 3 and w.upper() not in self.stopwords}, key=lambda w: (-len(w), w))
                for w in informative[:3]:
                    records.append((pos, f"{ctry}|{w}"))
            elif kind == "address_number":
                number = re.search(r"\b\d{2,8}\b", addr)
                if number and words:
                    informative = next((w for w in sorted(words, key=lambda w: (-len(w), w))
                                         if len(w) >= 3 and w.upper() not in self.stopwords), "")
                    if informative:
                        records.append((pos, f"{ctry}|{number.group()}|{informative}"))
        return pd.DataFrame(records, columns=["_pos", "_key"])

    def generate_candidate_pairs(self, df_ref: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        """Generate unique pairs from exact-name, informative-token and address-number blocks."""
        required = {"entity_id", "clean_name", "clean_address", "clean_country"}
        for label, frame in (("reference", df_ref), ("target", df_target)):
            if not required.issubset(frame.columns):
                raise ValueError(f"{label} data missing columns: {sorted(required - set(frame.columns))}")
        results = []
        for kind in ("exact_name", "name_token", "address_number"):
            ref = self._keys(df_ref, kind)
            target = self._keys(df_target, kind)
            if ref.empty or target.empty:
                continue
            rc, tc = ref._key.value_counts(), target._key.value_counts()
            safe = rc[rc <= self.max_group_size].index.intersection(tc[tc <= self.max_group_size].index)
            if len(safe) == 0:
                print(f"  -> {kind}: no blocks under group limit")
                continue
            ref_groups = {key: group._pos.to_numpy() for key, group in ref[ref._key.isin(safe)].groupby("_key", sort=False)}
            target_groups = {key: group._pos.to_numpy() for key, group in target[target._key.isin(safe)].groupby("_key", sort=False)}
            ref_ids, target_ids = df_ref.entity_id.to_numpy(), df_target.entity_id.to_numpy()
            pair_frames = []
            count = 0
            capped = False
            for key in sorted(safe):
                left_pos, right_pos = ref_groups[key], target_groups[key]
                limit = self.max_pairs_per_block - count
                if limit <= 0:
                    capped = True
                    break
                take = min(len(left_pos) * len(right_pos), limit)
                if take < len(left_pos) * len(right_pos):
                    capped = True
                left = np.repeat(left_pos, len(right_pos))[:take]
                right = np.tile(right_pos, len(left_pos))[:take]
                pair_frames.append(pd.DataFrame({"source1_entity_id": ref_ids[left],
                                                 "target_entity_id": target_ids[right]}))
                count += take
            if pair_frames:
                results.append(pd.concat(pair_frames, ignore_index=True))
            if capped:
                print(f"  -> {kind}: pair safety limit applied")
            print(f"  -> {kind}: {count:,} pairs from {len(safe):,} safe blocks")
        if not results:
            return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"])
        candidates = pd.concat(results, ignore_index=True).drop_duplicates(ignore_index=True)
        if self.max_candidates_per_entity:
            candidates = candidates.sort_values(["source1_entity_id", "target_entity_id"], kind="stable")
            candidates = candidates.groupby("source1_entity_id", sort=False).head(self.max_candidates_per_entity)
        print(f"  -> Total unique candidates: {len(candidates):,}")
        return candidates
