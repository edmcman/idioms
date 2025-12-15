"""Re-format the idioms dataset to use its non-compact form in a json format friendly for uptake by pyarrow.

Also requires a modification to the dataset to output the string form rather than the object-model-JSON form.
"""

from pathlib import Path
import argparse
import json
from typing import Optional, Any

from idioms.data.types import TypeInfo
from idioms.dataiter import MatchedFunctionDataset, MatchedBinaryDataset
from idioms.data.dataset import MatchedBinary, MatchedFunction

def to_json(fn, type2id: Optional[dict[TypeInfo, int]] = None) -> dict[str, Any]:
    # If given a MatchedBinary, return a binary-level JSON containing its matched functions
    if isinstance(fn, MatchedBinary):
        return {
            "binary_hash": fn.binary_hash,
            "repo": fn.repo,
            "call_graph": fn.call_graph,
            "unmatched": fn.unmatched,
            "matched_functions": [to_json(f, type2id) for f in fn.functions],
        }

    variable_types = {
        name: (typ.declaration("") if type2id is None else type2id[typ])
        for name, typ in fn.variable_types.items()
    }
    user_defined_types = [
        (typ.declaration("") if type2id is None else type2id[typ]) for typ in fn.user_defined_types
    ]
    return {
        "name": fn.name,
        "canonical_name": fn.canonical_name,
        "repo": fn.repo,
        "decompiled_code": fn.decompiled_code,
        "canonical_decompiled_code": fn.canonical_decompiled_code,
        "original_code": fn.original_code,
        "canonical_original_code": fn.canonical_original_code,
        # Ignore code tokens for now; we'll use just unigram tokenization
        # "memory_layout": {loc.json_key(): var.to_json() for loc, var in self.memory_layout.items()},
        "variable_types": variable_types,
        "return_type": fn.return_type._to_json(),
        "user_defined_types": user_defined_types,
        "binary_hash": fn.binary_hash,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()
    ds_path = Path(args.dataset)
    
    ds_class = MatchedBinaryDataset if args.binary else MatchedFunctionDataset

    train_set = ds_class(ds_path.glob("train*.tar"), shuffle=False)
    validation_set = ds_class(ds_path.glob("valid*.tar"), shuffle=False)
    test_set = ds_class(ds_path.glob("test*.tar"), shuffle=False)

    arrow = {}
    for name, partition in zip(["train", "valid", "test"], [train_set, validation_set, test_set]):
        arrow[name] = [to_json(fn) for fn in partition]
    
    filename = f"arrow-{ds_path.name}" + ("-binary" if args.binary else "-function") + ".json"
    with open(filename, "w") as fp:
        json.dump(arrow, fp, indent=2)

if __name__ == "__main__":
    main()
