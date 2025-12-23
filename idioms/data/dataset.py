"""Contains code to model training examples.

Some code modified from the replication package for DIRTY: Chen, Qibin, et al. "Augmenting decompiler output with learned variable names and types." 31st USENIX Security Symposium (USENIX Security 22). 2022.
"""
from typing import Dict, Optional, Set, Any, cast
import json
import itertools
import functools

from idioms.data.lexer import tokenize_raw_code
from idioms.data.function import CollectedFunction
from idioms.data.variable import Location, Variable, location_from_json_key
from idioms.data.types import TypeInfo, UDT, Disappear, TypeLibCodec


class DecompiledFunction:
    def __init__(
        self,
        name: str,
        code_tokens: list,
        source: dict[Location, Variable],
        target: dict[Location, Variable],
        valid: bool = True,
        raw_code: str = "",
        test_meta: Optional[Dict[str, Dict[str, bool]]] = None,
        binary: Optional[str] = None,
        ea: Optional[int] = None,
    ):
        self.name = name
        self.canonical_name: Optional[str] = None
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self._is_valid = valid # not really meaningful in idioms.
        self.raw_code = raw_code
        self.canonical_code: Optional[str] = None
        self.test_meta = test_meta
        self.binary = binary
        self.ea: Optional[int] = ea

    @classmethod
    def from_json(cls, d: Dict):
        source = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["source"].items()
        }
        target = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["target"].items()
        }
        return cls(
            d["name"],
            d["code_tokens"],
            source,
            target,
            test_meta=d.get("test_meta", None),
            binary=d.get("binary", None),
            ea=d.get("ea", None),
        )

    def to_json(self):
        assert self._is_valid
        source = {loc.json_key(): var.to_json() for loc, var in self.source.items()}
        target = {loc.json_key(): var.to_json() for loc, var in self.target.items()}
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
            "ea": self.ea,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, **kwargs):
        """Convert from a decoded CollectedFunction"""
        name = cf.debug.name # changed from .decompiler in the original DIRTY implementation.
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)

        source = {**cf.decompiler.local_vars, **cf.decompiler.arguments}
        target = {**cf.debug.local_vars, **cf.debug.arguments}

        # Remove variables that overlap on memory or don't appear in the code tokens
        source_code_tokens_set = set(code_tokens)
        target_code_tokens_set = set(tokenize_raw_code(cf.debug.raw_code))

        source = DecompiledFunction.filter(source, source_code_tokens_set)
        target = DecompiledFunction.filter(target, target_code_tokens_set, set(source.keys()))

        # Assign type "Disappear" to variables not existing in the ground truth
        varnames = set()
        for loc in source.keys():
            if loc not in target.keys():
                target[loc] = Variable(Disappear(), "", False)
        # Add special tokens to variables  to prevnt being sub-tokenized in BPE
        for var in source.values():
            varname = var.name
            varnames.add(varname)
        for idx in range(len(code_tokens)):
            if code_tokens[idx] in varnames:
                code_tokens[idx] = f"@@{code_tokens[idx]}@@"

        return cls(
            name,
            code_tokens,
            source,
            target,
            valid=(name == cf.debug.name and bool(source)), # we weren't using this anyway, but no longer as meaningful now that name == cf.debug.name is a tautology.
            binary=kwargs["binary"],
            raw_code=raw_code,
            ea=cf.ea,
        )

    @staticmethod
    def filter(
        mapping: dict[Location, Set[Variable]],
        code_tokens: Optional[Set[str]] = None,
        locations: Optional[Set[Location]] = None,
    ) -> dict[Location, Variable]:
        """Discard and leave these for future work:

        Multiple variables sharing a memory location (no way to determine ground truth);
        Variables not appearing in code (no way to get representation);
        Target variables not appearing in source (useless ground truth);
        """
        ret: dict[Location, Variable] = {}
        for location, variable_set in mapping.items():
            if len(variable_set) > 1:
                continue
            var = list(variable_set)[0]
            if code_tokens is not None and not var.name in code_tokens:
                continue
            if locations is not None and not location in locations:
                continue
            ret[location] = var
        return ret

    @property
    def is_valid_example(self):
        return self._is_valid
    

class MatchedFunction:
    """Contains information about a decompiled function and the corresponding original function
    """
    def __init__(self,
                 name: str,
                 canonical_name: str,
                 repo: str,
                 decompiled_code: str,
                 canonical_decompiled_code: str,
                 original_code: str,
                 canonical_original_code: str,
                 variable_types: dict[str, TypeInfo], # key: variable name, value: type of that variable
                 return_type: TypeInfo, # Taken from the original code
                 user_defined_types: list[UDT],
                 binary_hash: str,
                 function_decls: Optional[dict[str, str]] = None,
                 global_decls: Optional[dict[str, str]] = None,
                 ea: Optional[int] = None,
                 # There may not be a unique source hash
                 # source_hash: str, # preprocessed (e.g. gcc -E -P) source hash, not what you get directly from GitHub.
                ):
        self.name = name
        self.canonical_name = canonical_name
        self.repo = repo
        self.decompiled_code = decompiled_code
        self.canonical_decompiled_code = canonical_decompiled_code
        self.original_code = original_code
        self.canonical_original_code = canonical_original_code
        self.variable_types = variable_types
        self.return_type = return_type
        self.user_defined_types = user_defined_types
        self.binary_hash = binary_hash
        self.function_decls = function_decls
        self.global_decls = global_decls
        self.ea = ea

    def __eq__(self, other):
        attrs = [
            "name",
            "repo",
            "decompiled_code",
            "canonical_decompiled_code",
            "original_code",
            "canonical_original_code",
            "variable_types",
            "return_type",
            "user_defined_types",
            "binary_hash",
            "function_decls",
            "global_decls",
            "ea",
        ]
        if not isinstance(other, MatchedFunction):
            return False
        for a in attrs:
            if (orig := getattr(self, a)) != (ot := getattr(other, a)):
                # print(a)
                # if a == "variable_types":
                #     for (l_name, l_type), (r_name, r_type) in zip(orig.items(), ot.items()):
                #         print(l_name, l_type)
                #         print(r_name, r_type)
                #         print()
                return False
        return True
                

    def to_json(self, type2id: Optional[dict[TypeInfo, int]] = None) -> dict[str, Any]:
        # Can just to typ._to_json() but TypeLibCodec.encode excludes spaces, which helps save storage.
        variable_types = {
            name: (TypeLibCodec.encode(typ) if type2id is None else type2id[typ])
            for name, typ in self.variable_types.items()
        }
        user_defined_types = [
            (TypeLibCodec.encode(typ) if type2id is None else type2id[typ]) for typ in self.user_defined_types
        ]
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "repo": self.repo,
            "decompiled_code": self.decompiled_code,
            "canonical_decompiled_code": self.canonical_decompiled_code,
            "original_code": self.original_code,
            "canonical_original_code": self.canonical_original_code,
            "variable_types": variable_types,
            "return_type": self.return_type._to_json(),
            "user_defined_types": user_defined_types,
            "binary_hash": self.binary_hash,
            "function_decls": self.function_decls,
            "global_decls": self.global_decls,
            "ea": self.ea,
        }
    
    @classmethod
    def from_json(cls, d: dict[str, Any], id2type: Optional[dict[int, TypeInfo]] = None) -> "MatchedFunction":
        variable_types = {
            name: (cast(TypeInfo, TypeLibCodec.decode(typ)) if id2type is None else id2type[typ])
            for name, typ in d["variable_types"].items()
        } # typ will be an encoded JSON string if id2type is None; otherwise, it will be an integer.
        user_defined_types = [
            (cast(UDT, TypeLibCodec.decode(udt)) if id2type is None else cast(UDT, id2type[udt])) for udt in d["user_defined_types"]
        ]
        return MatchedFunction(
            name=d['name'],
            canonical_name=d['canonical_name'],
            repo=d['repo'],
            decompiled_code=d['decompiled_code'],
            canonical_decompiled_code=d['canonical_decompiled_code'],
            original_code=d["original_code"],
            canonical_original_code=d['canonical_original_code'],
            variable_types=variable_types,
            return_type=cast(TypeInfo, TypeLibCodec.decode(json.dumps(d['return_type']))),
            user_defined_types=user_defined_types,
            binary_hash=d['binary_hash'],
            function_decls=d.get('function_decls'),
            global_decls=d.get('global_decls'),
            ea=d.get('ea'),
        )
    

class MatchedBinary:
    def __init__(self, 
                 functions: list[MatchedFunction], 
                 binary_hash: str, 
                 repo: str, 
                 call_graph: dict[str, list[str]],
                 unmatched: dict[str, str] # decompiled functions for which we have no original code.
                ):
        self.binary_hash = binary_hash
        self.repo = repo
        self.functions = functions
        self.unmatched = unmatched 
        self.call_graph = call_graph

    def __eq__(self, other):
        return isinstance(other, MatchedBinary) and self.repo == other.repo and self.binary_hash == other.binary_hash and self.functions == other.functions and self.call_graph == other.call_graph and self.unmatched == other.unmatched
    
    @functools.cached_property
    def canonical_decompiled_code_lookup(self):
        lookup = self.unmatched.copy()
        for fn in self.functions:
            lookup[fn.name] = fn.canonical_decompiled_code
        return lookup

    def to_json(self, compact: bool=False) -> dict[str, Any]:
        """Convert this MatchedBinary to a json format. If compact=True, then variable types will
        be removed from the MatchedFunctions in this binary and stored in a separate typelib. This
        substantially decreases storage requirements, but it means that each MatchedFunction cannot
        be loaded separately from the enclosing MatchedBinary.
        """
        d: dict[str, Any] = {
            "binary_hash": self.binary_hash,
            "repo": self.repo,
            "call_graph": self.call_graph,
            "unmatched": self.unmatched
        }

        # Decrease storage requirements by associating each type with an ID and storing each full type definition exactly once.
        if compact:
            type2id: Optional[dict[TypeInfo, int]] = {}
            idsrc = 0
            for fn in self.functions:
                for typ in itertools.chain(fn.variable_types.values(), fn.user_defined_types):
                    if typ not in type2id:
                        type2id[typ] = idsrc
                        idsrc += 1
            d["typeids"] = [(ident, TypeLibCodec.encode(typ)) for typ, ident in type2id.items()]
        else:
            type2id = None
        
        d["matched_functions"] = [
            fn.to_json(type2id=type2id) for fn in self.functions
        ]
        
        return d
    
    @classmethod
    def from_json(cls, d: dict[str, Any]):
        if "typeids" in d:
            id2type: Optional[dict[int, TypeInfo]] = {
                ident: TypeLibCodec.decode(typ) # type: ignore
                for ident, typ in d["typeids"]
            }
        else:
            id2type = None
        
        functions = [
            MatchedFunction.from_json(fn_d, id2type=id2type) for fn_d in d["matched_functions"]
        ]

        return cls(
            functions=functions,
            binary_hash=d['binary_hash'],
            repo=d['repo'],
            call_graph=d['call_graph'],
            unmatched=d['unmatched']
        )