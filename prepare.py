"""Prepare a dataset for training, validation, and testing based on information from GHCC and DIRTY's dataset generator.
"""

import argparse
import json
import gzip
import multiprocessing
import tarfile
import io
import sys
import random
import itertools
import functools
from collections import deque
from os import PathLike, scandir
from pathlib import Path
from typing import NamedTuple, Optional, Iterator, Iterable, TypeVar
from typing import Union as tUnion

import tree_sitter_c
from tree_sitter import Node, Parser, Language
from tqdm import tqdm

from idioms.data.dataset import DecompiledFunction, MatchedFunction, MatchedBinary 
from idioms.data.function import CollectedFunction, MissingDebugError
from idioms.data.types import *

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("decompiled_dir", help="A path the location of the decompiled function info")
    parser.add_argument("metadata_info", help="Either a path to the compiled binaries or a JSON file containing the relevant information.")
    parser.add_argument("preprocessed_dir", help="A path to the location of the original code with preprocessor run")
    parser.add_argument("deduplication_file_or_repo_list", help="The json file containing deduplication information, or a list of repos if --single-split-name is specified.")
    parser.add_argument("output_dir", help="The directory into which the dataset should be written.")
    parser.add_argument("--single-split-name", type=str, help="Do not split into train/validation/test; instead, put all input repos in a partition of the specified name.")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes")
    parser.add_argument("--dataset-size", default=None, type=int, help="If set, specifies the maximum number of repos in the dataset")
    parser.add_argument("--holdout-set-size", default=0.1, type=float, help="The fraction of the overall repositories that should be included in the holdout set.")
    parser.add_argument("--random-seed", type=int, default=80, help="The random seed for this script.")
    parser.add_argument("--shard-size", type=int, default=3000, help="The number of examples per dataset shard.")
    parser.add_argument("--valid-max-bins-per-repo", type=int, help="The maximum number of binaries per repository in the validation set.")
    parser.add_argument("--test-max-bins-per-repo", type=int, help="The maximum number of binaries per repository in the test set.")
    return parser.parse_args()

##################################################
# Parse preprocessed C code & build examples
##################################################

def print_immediate_children(node: Node):
    for i, child in enumerate(node.children):
        print(f"{i}. {child.type}: {node.field_name_for_child(i)}: {child.text.decode()}")

def remove_curly_braces(nodes: list[Node]) -> list[Node]:
    assert nodes[0].type == "{" and nodes[-1].type == "}"
    return nodes[1:-1]

class TypeNotFoundError(Exception):
    """There is no type with this name."""
    pass

class TypeNotDefinedError(Exception):
    """An incomplete type (stub) exists, but its full definition does not."""
    pass

class UnsupportedFeatureError(Exception):
    """The specified code feature is not supported."""
    pass

# TODO: Make this customizable because the size of values can change depending on the platform.
# would also need to change the size value for Pointer.size and Enum.size
PRIMITIVE_TYPES = {
    "void": Void(),
    "char": TypeInfo(name="char", size=1),
    "unsigned char": TypeInfo(name="unsigned char", size=1),
    "short": TypeInfo(name="short", size=2),
    "unsigned short": TypeInfo(name="unsigned short", size=2),
    "int": TypeInfo(name="int", size=4),
    "unsigned int": TypeInfo(name="unsigned int", size=4),
    "long": TypeInfo(name="long", size=8),
    "unsigned long": TypeInfo(name="unsigned long", size=8),
    "long long": TypeInfo(name="long long", size=8),
    "unsigned long long": TypeInfo(name="unsigned long long", size=8),
    "float": TypeInfo(name="float", size=4),
    "double": TypeInfo(name="double", size=8),
    "long double": TypeInfo(name="long double", size=16),
    "_Bool": TypeInfo(name="_Bool", size=1)
}

def generate_primitive_aliases() -> dict[str, str]:
    """Standardize integer type names.
    
    There's more than one way to declare the same integer type in C. For example, 
    "unsigned long" is the same as "long unsigned". This function generates a dictionary that
    converts nonstandard names to the corresponding standard names.
    """
    def valid_with_extra_int(int_t: str) -> bool:
        """Some combinations of int type indentifiers are invalid, like
        int int, int char, signed char int, etc. This function identifiers them.
        """
        return all(invalid not in int_t for invalid in {"int", "char"})
    basic_ints = ["char", "short", "int", "long", "long long"]
    unsigned_ints = ["unsigned " + b for b in basic_ints]
    ints = basic_ints + unsigned_ints
    aliases: dict[str, str] = {}
    for int_t in basic_ints:
        aliases["signed " + int_t] = int_t
        aliases[int_t + " signed"] = int_t
    for int_t in basic_ints:
        aliases[int_t + " unsigned"] = "unsigned " + int_t

    for typename, simplified in itertools.chain(
        ((int_t, int_t) for int_t in ints if valid_with_extra_int(int_t)),
        [alias for alias in aliases.items() if valid_with_extra_int(alias[0])]
    ):
        aliases[typename + " int"] = simplified
    # Special cases that don't fit the patterns above
    aliases["unsigned"] = "unsigned int"
    aliases["signed"] = "int"
    aliases["double long"] = "long double"
    return aliases

PRIMITIVE_ALIASES = generate_primitive_aliases()

# A va_list's implementation is platform dependent. For x86-64, on which the dataset was built,
# the C definition for the implementation is:
# typedef struct {
#    unsigned int gp_offset;
#    unsigned int fp_offset;
#    void *overflow_arg_area;
#    void *reg_save_area;
# } va_list[1];
# The way the typedefs in the headers work out, __builtin_va_list is the "original" name of
# the type and va_list is an alias. We don't include the actual composition of the struct
# because this is an implementation detail that should NOT be relied upon in code.
BUILTINS = {
    "__builtin_va_list": TypeInfo(name="__builtin_va_list", size=24) # the size is 24 on x86-64, anyway.
}


class FileTypeMapping:
    ANONYMOUS_COMPOSITE_VALID_PARENTS = {"type_definition", "field_declaration", "declaration"}

    def __init__(self):
        # Maps the name from one type to another, as in a typedef.
        self.aliases: dict[str, str] = PRIMITIVE_ALIASES.copy()
        # Maps the textual name of a type (e.g. "char *") to its object-model representation 
        self.types: dict[str, TypeInfo] = PRIMITIVE_TYPES.copy()
        self.types.update(BUILTINS)
        # Maps the textual name of an incomplete type to a stub object which can be later mapped back to the original object.
        self.stubs: dict[str, TypeStub] = {}
        # Maps identifiers defined in enums to their values. For enums whose values are expressions, we don't evaluate the expressions and instead assign them to "None"
        self.enum_values: dict[str, Optional[int]] = {}
        # Builtin symbols for which we may or may not have a definition.
        self.builtins: set[str] = set(BUILTINS.keys())
        # Declared symbols (e.g. global variables, function declarations)
        self.declarations: dict[str, TypeInfo] = {}

    def add_alias(self, base_name: str, new_name: str):
        assert base_name in self.aliases or base_name in self.types or base_name in self.stubs or base_name in self.builtins, f"Aliasing an unknown symbol {base_name} to {new_name}"
        self.aliases[new_name] = base_name

    def add_stub(self, typ_name: str, stub: TypeStub):
        if typ_name in self.stubs:
            assert self.stubs[typ_name] == stub
        self.stubs[typ_name] = stub

    def add_type(self, typ_name: str, typ: TypeInfo):
        # if typ_name in self.stubs:
        #     del self.stubs[typ_name]
        assert not isinstance(typ, TypeStub), f"add_type should only be called on fully realized types but {typ} is a TypeStub."
        if isinstance(typ, (Struct, Union, Enum)):
            assert typ.name != ANONYMOUS_UDT_NAME, f"Cannot add anonymous type {typ} as a globally-available type."
        if typ_name in self.types:
            assert typ == self.types[typ_name], f"Conflicting declarations of type {typ_name}:\n{typ}\n  and\n{self.types[typ_name]}"
        else:
            self.types[typ_name] = typ

    def add_enum_value(self, name: str, value: Optional[int]):
        """Add record enum values so it is known what they are if they are used in the function.
        Note that :param value: can be None if the enum is initialized with an expression.
        """
        assert name not in self.enum_values
        self.enum_values[name] = value

    def add_declaration(self, typ: TypeInfo, name: str):
        if name in self.declarations:
            assert self.declarations[name] == typ, f"{name} has already been declared as a {self.declarations[name]} (attempting to redeclare as a {typ})."
        else:
            self.declarations[name] = typ

    def is_builtin(self, symbol: str):
        """Returns true if this symbol is a builtin, or an alias to one.
        """
        while symbol in self.aliases:
            symbol = self.aliases[symbol]
        return symbol in self.builtins

    def get_type(self, typ_name: str) -> Optional[TypeInfo]:
        """Get the type corresponding to this symbol
        """
        while typ_name in self.aliases:
            typ_name = self.aliases[typ_name]
        
        if typ_name in self.types:
            return self.types[typ_name]
        
        # assert typ is not None, f"Undefined symbol name without a type: {typ_name}"
        
        if typ_name in self.stubs:
            stub = self.stubs[typ_name]
            # assert stub.typ == typ, f"Inconsistent types for {typ_name}: {stub.typ} and {typ}"
            return stub
        return None

    def parse_type(self, node: Node) -> Optional[TypeInfo]:
        if node.type == "type_definition":
            return self.parse_typedef(node) # returns None
        if node.type == "struct_specifier":
            return self.parse_struct(node)
        if node.type == "union_specifier":
            return self.parse_union(node)
        if node.type == "enum_specifier":
            return self.parse_enum(node)
        if node.type == "declaration":
            return self.parse_declaration(node)

        return None
    
    def parse_typedef(self, node: Node):
        """Parse a member of a typedef into either an alias or a new type in the TypeInfo object model, depending on the composition of that typedef
        
        :param node: a "type_definition" node
        """
        typ_node = get_child(node, "type")
        declarator = get_child(node, "declarator")

        if typ_node.type in {"sized_type_specifier", "type_identifier", "primitive_type"}:
            typ_text = typ_node.text.decode()
            original_type = self.get_type(typ_text)
            if original_type is None and typ_node.type == "type_identifier":
                # HACK: check for builtins by checking that the builtin starts with a specific string.
                name = typ_node.text.decode()
                assert "__builtin" == name[:9] or self.is_builtin(name), f"Unknown identifier {name} is not a builtin."
                self.builtins.add(name)
                assert declarator.type == "type_identifier", f"Typedefs to non-identifier declarators are unsupported for builtins. declarator={declarator.type}, typedef={node.text.decode()}"
                new_name = declarator.text.decode()
                self.add_alias(name, new_name)
                return
        else:
            assert typ_node.type in {"struct_specifier", "enum_specifier", "union_specifier"}
            # NOTE: to change this to have pointers to user-defined types not expanded to their definitions, remove self.get_type here.
            # You'll have to follow the alias chain, however.
            original_type = self.get_type(typ_node.text.decode())
            if original_type is None or isinstance(original_type, TypeStub):
                original_type = self.parse_type(typ_node)
        assert original_type is not None, f"Unknown type node type, undeclared identifier, or anonymous type: {typ_node.type}: {typ_node.text.decode()}"

        if type(original_type) is TypeInfo and declarator.type in {"type_identifier", "primitive_type"}:
            assert original_type.name is not None, f"Type name for {original_type} in typedef {node.text.decode()} is None!"
            self.add_alias(original_type.name, declarator.text.decode())
        else:
            typ, name = self.parse_declarators(declarator, original_type)

            # In a typedef like
            #     typedef struct { int a; int b; } mystruct;
            # the struct object we get back from parse_type() will be anonymous and have a placeholder name.
            # The actual name is the name in the declarator.
            if isinstance(typ, (Struct, Union, Enum)) and original_type.name == ANONYMOUS_UDT_NAME:
                typ.name = name
            
            if isinstance(typ, TypeStub):
                self.add_stub(name, typ)
            else:
                self.add_type(name, typ)

    def _parse_member(self, node: Node) -> tUnion[UDT.Field, Struct, Union]:
        """Convert a member of a stucture or union in to Field object in the TypeInfo object model.
        If the field is an anonymous struct or union, return that struct or union instead.

        :param node: a "field_declaration" node
        """
        assert node.type == "field_declaration"
        type_node = get_child(node, "type")
        # NOTE: To change this so that pointers to composite types don't contain the definition of that type, remove the call to get_type here and only call parse_type.
        # This will make it so that structs/unions/enums are parsed into a stub form.
        # You'll have to follow the alias chain, however.
        base_type = self.get_type(type_node.text.decode())
        if base_type is None:
            base_type = self.parse_type(type_node)
        # TODO: If this type is defined as part of a local variable declaration, the type may actually
        # be present in a scope enclosing this one. In this case, we'll throw an exception, but we could
        # actually get the type. (However, this is rare.)
        if base_type is None:
            raise TypeNotFoundError(f"A type for {type_node.text.decode()} of struct/union member {node.text.decode()} is not defined.")

        declarator = node.child_by_field_name("declarator")
        if declarator is not None: # normal 'int x' or 'struct pt p' declaration.
            typ, field_name = self.parse_declarators(get_child(node, "declarator"), base_type)
            return UDT.Field(name=field_name, size=typ.size, type_name=typ)
        elif isinstance(base_type, (Union, Struct)): # An anonymous struct or union nested inside another struct or union
            assert base_type.name == ANONYMOUS_UDT_NAME, f"Non-anonymoust union member has no declarator in field declaration '{node.text.decode()}'"
            return base_type
        else:
            raise ValueError(f"Unrecognized field declaration format: no declarators and base type of type {type(base_type)} ({base_type})")

    def parse_struct(self, node: Node) -> tUnion[Struct, StructStub]:
        """Convert a node representing a structure into a TypeInfo object.

        :param node: a "struct_specifier" node.
        """
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type e.g. struct thing (with no { ... } defining the fields.)
            assert typ_identifier and typ_identifier.type == "type_identifier"
            struct_name = typ_identifier.text.decode()
            typ = StructStub(struct_name)
            self.add_stub("struct " + struct_name, typ)
        else:
            fields = []
            assert body.type == "field_declaration_list", f"Struct {node.text.decode()} has body of type {body.type}"
            for field in remove_curly_braces(body.children):
                fields.append(self._parse_member(field))
            if typ_identifier is None:
                assert node.parent and node.parent.type in FileTypeMapping.ANONYMOUS_COMPOSITE_VALID_PARENTS, f"Invalid parent for anonymous struct: {node.parent}"
                typ = Struct(name=ANONYMOUS_UDT_NAME, layout=fields)
            else:
                struct_name = typ_identifier.text.decode()
                typ = Struct(name=struct_name, layout=fields)
                self.add_type("struct " + struct_name, typ)
        
        return typ
    
    def parse_union(self, node: Node) -> tUnion[Union, UnionStub]:
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type e.g. union thing (with no { ... } defining the fields.)
            assert typ_identifier and typ_identifier.type == "type_identifier"
            union_name = typ_identifier.text.decode()
            typ = UnionStub(union_name)
            self.add_stub("union " + union_name, typ)
        else:
            fields = []
            assert body.type == "field_declaration_list", f"Union {node.text.decode()} has body of type {body.type}"
            for field in remove_curly_braces(body.children):
                fields.append(self._parse_member(field))
            if typ_identifier is None:
                assert node.parent and node.parent.type in FileTypeMapping.ANONYMOUS_COMPOSITE_VALID_PARENTS, f"Invalid parent for anonymous union: {node.parent}"
                typ = Union(name=ANONYMOUS_UDT_NAME, members=fields)
            else:
                union_name = typ_identifier.text.decode()
                typ = Union(name=union_name, members=fields)
                self.add_type("union " + union_name, typ)
        
        return typ

    def parse_enum(self, node: Node) -> Optional[tUnion[Enum, EnumStub]]:
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type
            assert typ_identifier and typ_identifier.type == "type_identifier"
            enum_name = typ_identifier.text.decode()
            typ = EnumStub(enum_name)
            self.add_stub("enum " + enum_name, typ)
        else:
            members = []
            value = 0 # values implicitly start at zero and are incremented, unless otherwise specified.
            assert body.type == "enumerator_list"
            for enumerator in remove_curly_braces(body.children):
                if enumerator.type == ",":
                    continue
                assert enumerator.type == "enumerator", f"Found non-enumerator type in enum!"
                value_node = enumerator.child_by_field_name("value")
                if value_node:
                    if value_node.type == "number_literal":
                        value = parse_int(value_node.text.decode())
                    else: # value is some expression. We could try to execute it to get the value, but that could be difficult and possibly dangerous
                        value = None
                enumerator_name = get_child(enumerator, "name").text.decode()
                members.append(Enum.Member(name=enumerator_name, value=value))
                self.add_enum_value(enumerator_name, value)
                if value is not None:
                    value += 1
            
            if typ_identifier is None: # It's an anonymous enum (possibly in a typedef, but renaming in that case will be handled down the stack in parse_typedef).
                typ = Enum(name=ANONYMOUS_UDT_NAME, members=members)
            else:
                enum_name = typ_identifier.text.decode()
                typ = Enum(name=enum_name, members=members)
                self.add_type("enum " + enum_name, typ)
        
        return typ
    
    def parse_declaration(self, node: Node):
        type_node = get_child(node, "type")
        base_type = self.get_type(type_node.text.decode())
        if base_type is None:
            base_type = self.parse_type(type_node)
        assert base_type is not None, f"No type for {type_node.text.decode()} found for declaration {node.text.decode()}."
        declarator = get_child(node, "declarator") # TODO: Update for multiple declarators.
        if declarator.type == "init_declarator": # We're only interested in the type for now.
            declarator = get_child(declarator, "declarator")
        typ, name = self.parse_declarators(declarator, base_type)
        self.add_declaration(typ, name)

    DECLARATOR_NODE_TYPES = {
        "identifier",
        "field_identifier",
        "type_identifier",
        "pointer_declarator",
        # "init_declarator",
        "array_declarator",
        "function_declarator",
        "parenthesized_declarator"
    }

    def parse_declarators(self, declarator: Node, base_type: TypeInfo) -> tuple[TypeInfo, str]:
        assert declarator.type in FileTypeMapping.DECLARATOR_NODE_TYPES, f"Unexpected declarator type: {declarator.type}: {declarator.text.decode()}"

        # if declarator.type == "init_declarator":
        #     # declarator.children[0] (declarator) - the name of the variable being declared, or a declarator for it
        #     # declarator.children[1] =
        #     # declarator.children[2] (value) - the expression used to initialize the variable.
        #     declarator = get_child(declarator, "declarator")

        # Pointer declarators can be nested arbitrarily deep (e.g. int ****** x).
        if declarator.type == "pointer_declarator":
            # param_declarator.children[0] (None) is an *
            # param_declarator.children[1] (declarator) is another declarator - possibly a pointer.
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, Pointer(target_type_name=base_type))
        
        if declarator.type == "array_declarator":
            # declarator.children[0]: (declarator) - another declarator
            # declarator.children[1]: [
            # declarator.children[2]: (size; optional) - the array size
            # declarator.children[3]: ]
            # size = int(get_child(declarator, "size").text.decode())
            size = get_array_size(declarator)
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, Array(nelements=size, element_size=base_type.size, element_type=base_type))
        
        if declarator.type == "function_declarator":
            # declarator.children[0]: (declarator)
            # declarator.children[1]: (parameters)
            parameters = self.parse_parameters(get_child(declarator, "parameters"))
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, FunctionType(return_type=base_type, parameters=parameters))

        if declarator.type == "parenthesized_declarator":
            assert declarator.children[0].type == "("
            assert declarator.children[2].type == ")"
            return self.parse_declarators(declarator.children[1], base_type) # declarator.children[1] is unnamed.
        
        assert declarator.type == "field_identifier" or declarator.type == "type_identifier" or declarator.type == "identifier"
        return base_type, declarator.text.decode()

    def parse_abstract_declarators(self, declarator: Node, base_type: TypeInfo) -> tuple[TypeInfo, Optional[str]]:
        if declarator.type == "abstract_pointer_declarator":
            descendant = declarator.child_by_field_name("declarator")
            typ = Pointer(base_type)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ)
        elif declarator.type == "abstract_array_declarator":
            descendant = declarator.child_by_field_name("declarator")
            size = get_array_size(declarator)
            typ = Array(nelements=size, element_size=base_type.size, element_type=base_type)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ)
        elif declarator.type == "abstract_function_declarator":
            descendant = declarator.child_by_field_name("declarator")
            parameters = self.parse_parameters(get_child(declarator, "parameters"))
            typ = FunctionType(return_type=base_type, parameters=parameters)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ)
        elif declarator.type == "abstract_parenthesized_declarator":
            assert declarator.children[0].type == "(" and declarator.children[2].type == ")"
            return self.parse_abstract_declarators(declarator.children[1], base_type)

        return self.parse_declarators(declarator, base_type)

    def parse_parameters(self, param_list: Node) -> list[tuple[TypeInfo, Optional[str]]]:
        assert param_list.type == "parameter_list"
        assert param_list.children[0].type == "(" and param_list.children[-1].type == ")"
        parameters: list[tuple[TypeInfo, Optional[str]]] = []
        for param in param_list.children[1:-1]:
            if param.type == ",":
                continue
            if param.type == "variadic_parameter":  # variable number of arguments, denoted ...
                parameters.append((FunctionType.VariadicParameter(), None))
                continue
            # ANSI C parameters are of type parameter_declaration
            if param.type == "identifier":
                raise UnsupportedFeatureError("K&R-style parameter lists are not supported.")
            typ_node = get_child(param, "type")
            base_type = self.get_type(typ_node.text.decode())
            if base_type is None:
                base_type = self.parse_type(typ_node)
            if base_type is None:
                raise TypeNotFoundError(f"Parameter declaration with unknown type: {param.text.decode()}: {typ_node.text.decode()}")
            declarator = param.child_by_field_name("declarator")
            if declarator is None:
                parameters.append((base_type, None))
            else:
                parameters.append(self.parse_abstract_declarators(declarator, base_type))

        return parameters
    
    def __str__(self):
        components = ["FileTypeMapping:", "  Aliases:"]
        components.extend(
            f"    {alias}: {orig}"
            for alias, orig in self.aliases.items()
        )
        components.append("  Types:")
        components.extend(
            f"    {name}: {typ}"
            for name, typ in self.types.items()
        )
        components.append("  Incomplete Types:")
        components.extend(
            f"    {name}: {stub}"
            for name, stub in self.stubs.items()
        )
        components.append("  Enum Values:")
        components.extend(
            f"    {name}=" + ("<expr>" if value is None else str(value))
            for name, value in self.enum_values.items()
        )
        components.append("  Builtins:")
        components.extend(
            f"    {name}" for name in self.builtins
        )
        components.append("  Declarations:")
        components.extend(
            f"    {name}: {typ}"
            for name, typ in self.declarations.items()
        )
        return "\n".join(components)

def get_child(node: Node, child: tUnion[int, str]) -> Node:
    """A wrapper around .child_by_field_name and .children[] that fails with an exception
    where there is no such child.
    """
    if isinstance(child, str):
        ret = node.child_by_field_name(child)
        assert ret is not None, f"{node.type} has no child named {child}: {', '.join(f'{c.type}: {c.grammar_name}' for c in node.children)}"
    else:
        ret = node.children[child]
    return ret

def get_array_size(declarator: Node) -> int:
    """Get the array size from an array_declarator or abstract_array_declarator
    """
    size_node = declarator.child_by_field_name("size")
    if size_node:
        if size_node.type == "number_literal":
            size = parse_int(size_node.text.decode())
        else: # There's some kind of expression determining the size.
            size = -1
    else: # This is possible with a flexible array member in a struct. They have no inherent size; the extra space must be allocated dynamically.
        size = 0
    return size

def parse_int(s: str) -> int:
    """Return an integer value that corresponds to the string.
    """
    s = s.lower()
    while s[-1] == "u" or s[-1] == "l":
        s = s[:-1]
    if s[:2] == "0x":
        return int(s, base=16)
    else:
        return int(s)

### Full example of an ambiguous typedef resulting in an ERROR node
# typedef __signed__ char __s8;
# 0. typedef: None: typedef
# 1. type_identifier: type: __signed__
# 2. primitive_type: declarator: char
# 3. ERROR: None: __s8
# 4. ;: None: ;
# __signed__ is a compiler-defined type qualifier; tree-sitter isn't aware of this.
# Instead, it assumes that __signed__ is the type being declared and that char is its
# alias. Then __s8 is an invalid extra token: an ERROR node.
def has_error(node: Node) -> bool:
    """Return True if there is an ERROR node in the subtree rooted at `node`.
    
    Even for C code that compiles, there are occasionally parts of code that tree-sitter
    fails to parse. This is often due to implementation-specific details; for instance,
    typedef __signed__ char __s8; fails to parse because __signed__ is not declared and so
    tree-sitter must make its best guess.
    """
    return node.type == "ERROR" or any(has_error(c) for c in node.children)

def contains_node_of_types(node: Node, types: set[str]) -> bool:
    """Determine if this node or one of its children is of one of the provided types."""
    return node.type in types or any(contains_node_of_types(c, types) for c in node.children)

def find_types(root: Node) -> FileTypeMapping:
    """Find all defined and declared types in the immediate children of `root`.

    :param root: an AST node, intended to be of type "translation_unit"
    :returns: an object containing information about type-related symbols.
    """
    preproc_types = FileTypeMapping()
    for member in root.children:
        if not has_error(member):
            try:
                preproc_types.parse_type(member)
            except (AssertionError, ValueError, TypeNotFoundError):
                pass

    return preproc_types


class Scope:
    def __init__(self, *, mapping: Optional[FileTypeMapping] = None, enclosing: Optional["Scope"] = None):
        """Create a scope in C code. Can be initialized with a library of existing types in the scope and
        can be nested inside another scope.

        :param mapping: A library of existing types in the scope. If there are none, pass "None"; a FileTypeMapping object will be created lazily as needed.
        :param enclosing: The scope in which this scope exists. This should be "None" for the global scope.
        """
        self.mapping = mapping
        self.enclosing = enclosing

    def _get_type(self, type_text: str) -> Optional[TypeInfo]:
        """Check this scope, then enclosing scopes recursively for the type.

        :param type_text: the text describing this type (i.e. from node.text.decode())
        :returns: the type if it is defined at this scope or an enclosing scope; None otherwise.
        """
        typ = None
        if self.mapping is not None:
            typ = self.mapping.get_type(type_text)
        if typ is None and self.enclosing is not None:
            typ = self.enclosing._get_type(type_text)
        return typ
    
    def get_or_parse_type(self, node: Node) -> TypeInfo:
        """If this type exists at this scope or an enclosing scope, return
        that type. If not, parse it and add it to this scope.

        :param node: a node that describes a type. Raise an exception if this node does not correspond to a recognized type.
        :returns: the corresponding TypeInfo object.
        """
        typ = self._get_type(type_text = node.text.decode())
        if typ is None:
            if self.mapping is None:
                self.mapping = FileTypeMapping()
            typ = self.mapping.parse_type(node)
            if typ is None:
                raise UnsupportedFeatureError(f"Cannot parse node {node.text.decode()} (type: {node.type}) into a TypeInfo object.")
        return typ
    
    def parse_declarators(self, declarator: Node, base_type: TypeInfo) -> tuple[TypeInfo, str]:
        if self.mapping is None:
            self.mapping = FileTypeMapping()
        try:
            return self.mapping.parse_declarators(declarator, base_type)
        except TypeNotFoundError:
            # This happens when function pointers include arguments of a type that is not declined at the current
            # scope (but may be defined at an enclosing scope.) In this case, we try the enclosing scope.
            #
            # This strategy will fail when there are arguments of types from multiple different scopes present as
            # arguments to this function pointer. The Scope class can handle this with the _get_type method that 
            # searches enclosing scopes for types, though FileTypeMapping represents a mapping for a single scope.
            # Because, in the vast majority of cases, types are defined at the global scope, we simply re-attempt to parse
            # the function at an enclosing scope.
            #
            # Error out if a new type is defined in the arguments of a parameter list; we don't want to polute the enclosing scope's namespace.
            if self.enclosing is None or contains_node_of_types(declarator, {"field_declaration_list", "enumerator_list"}):
                raise
            else:
                return self.enclosing.parse_declarators(declarator, base_type)

    
    def parse_abstract_declarators(self, declarator: Node, base_type: TypeInfo) -> tuple[TypeInfo, str | None]:
        if self.mapping is None:
            self.mapping = FileTypeMapping()
        return self.mapping.parse_abstract_declarators(declarator, base_type)
    
    # Perhaps move this to FileTypeMapping.
    def expand_type(self, typ: TypeInfo) -> TypeInfo:
        """Expand incomplete type stubs into full types, except in the case of recursively defined types.
        This method treats types as immutable, and the input may not be the same object as the output.

        :param typ: a type to expand.
        :returns: the type, expanded.
        """
        if type(typ) is TypeInfo:
            return typ
        
        # Keep track of which user-defined types we've encountered before.
        # If we encounter the same one, that means we've hit a recursive data structure or 
        # the same component type is used multiple times within a type.
        encountered: set[TypeStub] = set()

        def expand(stub: TypeStub):
            typ = self._get_type(str(stub))
            if typ is None or isinstance(typ, TypeStub):
                raise TypeNotDefinedError(f"No definition for type {typ}")
            return typ

        def explore(typ: TypeInfo) -> TypeInfo:
            if isinstance(typ, TypeStub) and typ not in encountered:
                typ = expand(typ)
            # We never add ANONYMOUS_UDT_NAME to encountered if typ is an anonymoust struct or union typ.stub always returns False.
            elif isinstance(typ, (Struct, Union)) and typ.stub in encountered:
                typ = typ.stub

            if isinstance(typ, Struct):
                if typ.name != ANONYMOUS_UDT_NAME:
                    encountered.add(typ.stub)
                layout = []
                for field in typ.layout:
                    if isinstance(field, UDT.Field):
                        assert isinstance(field.type_name, TypeInfo)
                        field_type = explore(field.type_name)
                        layout.append(UDT.Field(name=field.name, size=field_type.size, type_name=field_type))
                    else:
                        assert isinstance(field, (Struct, Union)) 
                        layout.append(explore(field))
                typ = Struct(name=typ.name, layout=layout)
            if isinstance(typ, Union):
                if typ.name != ANONYMOUS_UDT_NAME:
                    encountered.add(typ.stub)
                members = []
                for field in typ.members:
                    if isinstance(field, UDT.Field):
                        assert isinstance(field.type_name, TypeInfo)
                        field_type = explore(field.type_name)
                        members.append(UDT.Field(name=field.name, size=field_type.size, type_name=field_type))
                    else:
                        assert isinstance(field, (Struct, Union))
                        members.append(explore(field))
                typ = Union(name=typ.name, members=members)
            elif isinstance(typ, Pointer):
                assert isinstance(typ.target_type_name, TypeInfo)
                expanded = explore(typ.target_type_name)
                if expanded is not typ.target_type_name:
                    typ = Pointer(expanded)
            elif isinstance(typ, Array):
                assert isinstance(typ.element_type, TypeInfo)
                expanded = explore(typ.element_type)
                if expanded is not typ.element_type:
                    typ = Array(nelements=typ.nelements, element_size=typ.element_size, element_type=expanded)
            elif isinstance(typ, FunctionType):
                typ = FunctionType(
                    return_type = explore(typ.return_type),
                    parameters = [
                        (explore(param_type), param_name)
                        for param_type, param_name in typ.parameters
                    ]
                )
            return typ
    
        return explore(typ)


def add_storage_class_specifiers(subnode: Node, decls: list[bytes]) -> list[bytes]:
    """Add storage class specifiers to canonical declarations from the original declaration if one exists.
    """
    for child in reversed(subnode.children):
        if child.type == "storage_class_specifier":
            decls = [child.text + b" " + d for d in decls]
    return decls

class PreprocessedFunction:
    """A package of information about a preprocessed function.
    """
    def __init__(self, node: Node, file_types: FileTypeMapping):
        """
        :param node: the tree-sitter AST node that corresponds to this function definition
        :param file_types: the types defined for this function.
        """
        self.node = node
        self.file_types = file_types
        type_text = get_child(node, "type").text.decode()
        base_type = file_types.get_type(type_text)
        # Can either just raise an exception or parse the type here. If parsing the type, be careful that it isn't added to `types`
        if base_type is None:
            raise TypeNotFoundError(f"Return type {type_text} not found.")
        typ, name = file_types.parse_declarators(get_child(node, "declarator"), base_type)
        assert isinstance(typ, FunctionType) and typ.parameters is not None, f"Invalid declarator for a function definition: {typ}"
        assert all(p[1] is not None or isinstance(p[0], FunctionType.VariadicParameter) for p in typ.parameters), f"Function {name} has parameters with no name:\n{node.text.decode()}"
        self.typ: FunctionType = typ
        self.name: str = name
        self.text = self.node.text.decode()
        self.canonical_text, self.variable_types, self.return_type, self.referenced_types, self.referenced_identifiers = self._get_canonical_form()

    def _get_canonical_form(self) -> tuple[str, dict[str, TypeInfo], TypeInfo, list[TypeInfo], list[str]]:
        """Get the text of this function, all types for all variables in this function, the return type, and all other types 
        and identifiers referenced by this function in canonical form.
        """
        global_scope = Scope(mapping=self.file_types)

        # This canonicalizes both the return type and the parameters' types.
        canonical_function_type = global_scope.expand_type(self.typ)
        assert isinstance(canonical_function_type, FunctionType)

        fn_types: dict[str, TypeInfo] = {
            p_name: p_type # No type expansion necessary because this has already been done by expanding the overall function type above.
            for p_type, p_name in canonical_function_type.parameters if p_name is not None
        }
        assert len(fn_types) == len(self.typ.parameters) or (len(fn_types) + 1 == len(self.typ.parameters) and isinstance(self.typ.parameters[-1][0], FunctionType.VariadicParameter)), \
              f"All parameters to a function definition should have a name: {self.typ}"
        
        return_type = canonical_function_type.return_type # Expanding this is not necessary because it's been done as part of the expansion of th overall function type above.
        
        # Represents modifications that must be made to this function to canonicalize it.
        # Each entry in the list represents one modification. The first entry in the list
        # is the AST node for which the corresponding text is removed. The second entry represents
        # the replacement. If the replacement is given in bytes, Node is treated as an expression
        # and no semicolons are added. If the replacement is in a list, it is treated as (a) declaration
        # statement(s) and semicolons are added to each one. Additionally, the node is scanned for 
        # storage class specifiers, which are added back to each declaration, if present.
        #
        # To generalize this to arbitrary statements/expressions, the storage-class-specifier-scanning
        # functionality needs to be refactored into this function from _canonicalize_text
        edits: list[tuple[Node, bytes | list[bytes]]] = []

        # Types that are referenced in the function but do not correspond to a particular variable (e.g. in a typecast or sizeof expression).
        # We want a set, but consistent ordering, so we use a dictionary from Python 3.7+.
        # A consistent ordering is more predictable (potentially less confusing to an ML model) 
        # and a set reduces memory consumption; some TypeInfo representations can be quite large.
        referenced_types: dict[TypeInfo, None] = {}
        
        referenced_identifiers: set[str] = set()

        body = get_child(self.node, "body")

        def find_declarators(node: Node, scope: Scope):
            if node.type == "for_statement" or node.type == "compound_statement":
                scope=Scope(enclosing=scope)
            if node.type == "declaration":
                declared: list[bytes] = []
                base_type = scope.get_or_parse_type(get_child(node, "type"))
                for declarator in node.children_by_field_name("declarator"):
                    if declarator.type == "init_declarator":
                        initial_value = get_child(declarator, "value")
                        declarator = get_child(declarator, "declarator")
                    else:
                        initial_value = None
                    typ, name = scope.parse_declarators(declarator, base_type)
                    canonical_type = scope.expand_type(typ) # Put the type in a canonical form.

                    if name in fn_types: # Ensure that all declarations of the same variable in the same function are the same type.
                        # Failing this assertion is still technically legal in C, but is very uncommon and also bad coding practice.
                        # We could accomodate this, but it would increase the complexity of differentiating variables for little benefit.
                        # Instead, we'll just raise an exception and filter these functions out.
                        if canonical_type != fn_types[name]:
                            raise TypeNotFoundError(f"Variable {name} is declared twice (possibly at different scopes) with different types: {canonical_type} and {fn_types[name]}")
                    else:
                        fn_types[name] = canonical_type
                    canonical_declaration = bytes(canonical_type.stubify().declaration(name), "utf8")
                    if initial_value is None:
                        declared.append(canonical_declaration)
                    else:
                        declared.append(canonical_declaration + b" = " + initial_value.text)
                # Record declaration AST nodes and the corresponding canonical declarations so that we can use them to canonicalize the code of the function later.
                edits.append((node, declared))
            elif node.type == "type_descriptor":
                # Prepare the edit for canonicalizing type descriptors. Doing this requires the canonical type.
                base_type = scope.get_or_parse_type(get_child(node, "type"))
                declarator = node.child_by_field_name("declarator")
                if declarator is None:
                    typ = base_type
                else:
                    typ, name = scope.parse_abstract_declarators(declarator, base_type)
                    assert name is None, f"Expected abstract declarator with no name but found {name} in {node.text.decode()}"
                canonical_type = scope.expand_type(typ)
                referenced_types[canonical_type] = None
                edits.append((node, bytes(canonical_type.stubify().declaration(""), "utf8")))
            elif node.type == "identifier":
                referenced_identifiers.add(node.text.decode())
            else:
                for child in node.children:
                    find_declarators(child, scope)

        find_declarators(body, global_scope)

        canonical_text = self._canonicalize_text(bytes(canonical_function_type.stubify().declaration(self.name), "utf8"), edits)

        return canonical_text, fn_types, return_type, list(referenced_types), list(referenced_identifiers)
    
    def _canonicalize_text(self, function_declaration: bytes, body_edits: list[tuple[Node, bytes | list[bytes]]]) -> str:
        """Convert declarations to their canonical form, then return the text of the entire function.

        :param function_declaration: the canonical declaration for this function, in bytes.
        :param body_edits: changes to make to the function body to canonicalize it.
        """
        fn_declarator_node = get_child(self.node, "declarator")

        # Ensure edits are in sorted (descending) order, which makes the implementation much easier.
        # Editing the function at a point means the offsets for all subsequent points need to be adjusted.
        # By editing in descending order, there are no subsequent points.
        body_edits.sort(key=lambda decl: decl[0].start_byte, reverse=True)
        # Ensure nodes are non-overlapping.
        assert all(a[0].start_byte > b[0].end_byte for a, b in zip(body_edits, itertools.islice(body_edits, 1, None)))
        if len(body_edits) > 0:
            assert fn_declarator_node.end_byte < body_edits[-1][0].start_byte

        fn_start = self.node.start_byte

        text = self.node.text
        components = []
        for subnode, replacement in body_edits:
            components.append(text[(subnode.end_byte - fn_start):])
            if isinstance(replacement, list):
                components.append(b"; ".join(add_storage_class_specifiers(subnode, replacement)) + b";")
            else:
                components.append(replacement)
            text = text[:(subnode.start_byte - fn_start)]

        # Handle the declaration of this function separately. No semicolons should be added for the 
        # function declaration, and it spans two nodes (the base_type_node and the declarator_node), unlike
        # variable declarations, which are captured in a single node.
        components.append(text[(fn_declarator_node.end_byte - fn_start):])
        components.extend(add_storage_class_specifiers(self.node, [function_declaration]))

        components.reverse() # We've been adding components backwards, reverse them for the correct output.
        return b"".join(components).decode("utf8")

    def __str__(self):
        # Prints the function header.
        return get_child(self.node, "declarator").text.decode()


def find_functions(root: Node, types: FileTypeMapping) -> list[PreprocessedFunction]:
    functions: list[PreprocessedFunction] = []
    for member in root.children:
        if member.type == "function_definition" and not has_error(member):
            try:
                functions.append(PreprocessedFunction(member, types))
            except (TypeNotFoundError, TypeNotDefinedError, UnsupportedFeatureError, UnicodeDecodeError):
                pass # traceback.print_exc()
            except AssertionError:
                # Represents an assumption that was not met.
                pass # TODO: Handle this differently.

    return functions

def parse_file(file: PathLike) -> Node:
    """Parse the contents of a C file with tree-sitter.

    :param file: the C file.
    :returns: the root node of the AST generated from that file.
    """
    with open(file, "rb") as fp:
        contents = fp.read()
    tree = parser.parse(contents)
    return tree.root_node


def read_decompiled(location: Path, binary: str) -> Optional[list[DecompiledFunction]]:
    examples = []
    file = f"{binary}_{binary}.jsonl.gz"
    try:
        with gzip.open(location / file, "rt") as fp:
            for line in fp:
                cf = CollectedFunction.from_json(json.loads(line))
                examples.append(DecompiledFunction.from_cf(cf, binary=binary, max_stack_length=1024, max_type_size=1024))    
    except (gzip.BadGzipFile, EOFError):
        print(f"Bad gzip file: {file}")
        return None
    except MissingDebugError:
        print(f"Missing debug info in {file}")
        return None
    return examples

def canonicalize_function_names(functions: list[DecompiledFunction]) -> dict[str, list[str]]:
    """Remove function names from decompiled code and replace them with generic placeholders 
    funcX where X is an integer. Occasionally, the hex-rays-reported name of a function does not match
    the name of the function in the decompilation. In these cases, we correct the DecompiledFunction's
    names to match the decompilation.

    This is done on a binary level so that the same function name assigned the same placeholder 
    throughout the binary. The functions provided are assumed to all be a part of the same binary.

    In completing this process, this function essentially computes a call graph. That call graph is
    returned.
    """
    # Don't initialize with {bytes(fn.name, "utf8"): i for i, fn in enumerate(functions)}.
    # This is for two reasons:
    # 1. Sometimes the .name attribute of a DecompiledFunction is different from what's actually in the code.
    # 2. If the function fails to parse, then we'll have what is essentially a dangling reference to that function in the call graph
    #    (we won't have canonical code for that function but there will be a reference to it in the graph.)
    # Both of these are rare, but can happen.
    name2id: dict[bytes, int] = {} 

    def make_edit(node: Node) -> tuple[Node, bytes]:
        if node.text in name2id:
            return (node, bytes(f"func{name2id[node.text]}", "utf8"))
        else:
            newid = len(name2id)
            name2id[node.text] = newid
            return (node, bytes(f"func{newid}", "utf8"))

    state: list[tuple[DecompiledFunction, Node, list[tuple[Node, bytes]]]] = []
    
    for fn in functions:
        # First, find a reference to the function definition itself.
        root = parser.parse(bytes(fn.raw_code, "utf8")).root_node
        for fn_node in root.children:
            if fn_node.type == "comment":
                continue
        if fn_node.type != "function_definition": #, f"Expected function definition but found {fn_node}."
            continue

        declarator = get_child(fn_node, "declarator")
        while declarator.type != "identifier":
            try:
                declarator = get_child(declarator, "declarator")
            except:
                break
        
        if declarator.type != "identifier":
            continue
        
        # Record the names of the functions found in the binary.
        if fn.name == 'main':
            edits: list[tuple[Node, bytes]] = []
            fn.canonical_name = 'main'
        else:
            edit = make_edit(declarator)
            edits: list[tuple[Node, bytes]] = [edit]
            if edit[0].text.decode() != fn.name:
                fn.name = edit[0].text.decode()
            fn.canonical_name = edit[1].decode()
        
        state.append((fn, fn_node, edits))

    # Accumulate the call graph, since it's basically computed here anyway.
    call_graph: dict[str, list[str]] = {}

    # Edit the function to remove the names of all functions found in the binary.
    for fn, fn_node, edits in state:
        def find_fn_names(node: Node):
            if node.type == "call_expression":
                name = get_child(node, "function")
                # name.type == identifier: could also be a dereferenced function pointer, which we'll ignore because it's not a function name.
                # name.text in name2id: we only want to canonicalize function names that are actually in the binary.
                if name.type == "identifier" and name.text in name2id and name.text != b'main':
                    edits.append(make_edit(name))
            else:
                for child in node.children:
                    find_fn_names(child)
        
        find_fn_names(fn_node)

        ### Build the call graph from the edits before we flip the order of 'edits'.
        # "edits" also serves as the list of function calls in the call graph
        repeated = set() # don't add the same function twice to the call graph.
        called = []
        for call_node, _ in edits:
            if call_node.text not in repeated:
                called.append(call_node.text.decode())
                repeated.add(call_node)
        call_graph[fn.name] = called

        ### Edit the function to remove the function names.
        # As with PreprocessedFunction, reversing the order in which we apply the edtis means we
        # have less bookkeeping to do.
        edits.sort(key=lambda e: e[0].start_byte, reverse=True)
        fn_start = fn_node.start_byte
        text = fn_node.text
        components = []
        for subnode, replacement in edits:
            components.append(text[(subnode.end_byte - fn_start):])
            components.append(replacement)
            text = text[:(subnode.start_byte - fn_start)]
        components.append(text)
        components.reverse()
        canonical_code = b"".join(components).decode("utf8")
        fn.canonical_code = canonical_code
    
    return call_graph

def get_all_user_defined_types(original_fn: PreprocessedFunction) -> list[UDT]:
    """Return the definitions of all named user-defined types (UDTs), with the definitions of 
    all UDTs referenced in a definition reduced to incomplete types and listed separately.

    :original_fn: a PreprocessedFunction
    :returns: all user-defined types 
    """
    worklist: deque[TypeInfo] = deque(itertools.chain(original_fn.variable_types.values(), original_fn.referenced_types))
    worklist.append(original_fn.return_type)

    udts: list[UDT] = []
    already_defined: set[UDT] = set() # no need to define the same type twice.

    def _depth1_stubify(typ: Struct | Union) -> Struct | Union:
        fields = []
        for field in (typ.layout if isinstance(typ, Struct) else typ.members):
            if isinstance(field, (Struct, Union)):
                fields.append(field.stubify())
                worklist.append(field)
            elif isinstance(field, UDT.Field):
                assert isinstance(field.type_name, TypeInfo), f"All types from preprocessed source code should be represented in terms of TypeInfo objects."
                fields.append(UDT.Field(name=field.name, size=field.size, type_name=field.type_name.stubify()))
                worklist.append(field.type_name)
            else:
                fields.append(field)
        if isinstance(typ, Struct):
            return Struct(name=typ.name, layout=fields)
        else:
            return Union(name=typ.name, members=fields, padding=typ.padding)
        
    while len(worklist) > 0:
        typ = worklist.popleft()
        if type(typ) is TypeInfo:
            continue
        if isinstance(typ, (Struct, Union)):
            reduced = _depth1_stubify(typ)
            if reduced not in already_defined and reduced.name != ANONYMOUS_UDT_NAME:
               already_defined.add(reduced)    
               udts.append(reduced)  
        elif isinstance(typ, Pointer):
            assert isinstance(typ.target_type_name, TypeInfo)
            worklist.append(typ.target_type_name)
        elif isinstance(typ, Array):
            assert isinstance(typ.element_type, TypeInfo)
            worklist.append(typ.element_type)
        elif isinstance(typ, FunctionType):
            worklist.append(typ.return_type)
            worklist.extend(t for t, _ in typ.parameters if not isinstance(t, FunctionType.VariadicParameter))
        elif isinstance(typ, Enum) and typ not in already_defined and typ.name != ANONYMOUS_UDT_NAME:
            udts.append(typ)
            already_defined.add(typ)

    return udts


def build_matched_function(decompiled_fn: DecompiledFunction, original_fn: PreprocessedFunction, repo: str) -> MatchedFunction | None:
    """Use information about a DecompiledFunction and a PreprocessedFunction to create a single input/output pair
    for ML training and evaluation.
    """
    assert decompiled_fn.name == original_fn.name
    assert decompiled_fn.binary is not None
    # Canonical name and code are required for training/evaluation so fail if they're missing.
    if decompiled_fn.canonical_name is None or decompiled_fn.canonical_code is None:
        return None

    udts = get_all_user_defined_types(original_fn)
    # Extract function and global declarations for inclusion in the MatchedFunction
    function_decls: dict[str, str] = {}
    global_decls: dict[str, str] = {}

    _, _, _, _, referenced_identifiers = original_fn._get_canonical_form()

    used_declarations = {name: typ for name, typ in original_fn.file_types.declarations.items() if name in referenced_identifiers}

    for name, typ in used_declarations.items():
        try:
            decl_text = typ.stubify().declaration(name)
        except Exception:
            # If we cannot produce a declaration for whatever reason, fall back to the raw declaration text in the file (if any) or the name alone.
            decl_text = name
        if isinstance(typ, FunctionType):
            function_decls[name] = decl_text
        else:
            global_decls[name] = decl_text
    
    return MatchedFunction(
        name=decompiled_fn.name,
        canonical_name=decompiled_fn.canonical_name,
        repo=repo,
        decompiled_code=decompiled_fn.raw_code,
        canonical_decompiled_code=decompiled_fn.canonical_code,
        original_code=original_fn.text,
        canonical_original_code=original_fn.canonical_text,
        variable_types=original_fn.variable_types,
        return_type=original_fn.return_type,
        user_defined_types=udts,
        binary_hash=decompiled_fn.binary,
        function_decls=function_decls,
        global_decls=global_decls,
        ea=decompiled_fn.ea,
    )

K = TypeVar("K")
V = TypeVar("V")
def accumulate(d: dict[K, list[V]], key: K, value: V):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]

def prepare_repository(repo: str, binaries: set[str], preprocessed_location: Path, decompiled_location: Path) -> list[MatchedBinary]:
    """Prepare all examples in a repository for training. Combines decompiled information from IDA with preprocessed information.

    :param repo: the repository to process
    :param binaries: the hashes of all of the binaries in the compiled version of this program
    :param preprocessed_location: where to find the preprocessed files
    :param decompiled_location: where to find the decompiled files and typelib
    """
    bin2decomp: dict[str, list[DecompiledFunction]] = {} # Index by binary hash
    for binary in sorted(binaries):
        decomps = read_decompiled(decompiled_location, binary)
        if decomps: # ignore both empty list and None:
            nonstub = [d for d in decomps if d.name[0] != '.'] # Filter out PLT stub functions, which are prefixed with '.' by the DIRTY generator.
            if len(nonstub) > 0:
                bin2decomp[binary] = nonstub
    
    ### Eliminate object files if one or more binaries built from those object files are present.
    # Object files are considered "binaries", and are included in the dataset. This is desirable when
    # a project does not build completely; some data can still be salvaged from it. However, when the
    # binary built from those object files is present, the object files will necessarily contain
    # duplicate functions.
    bin2names: dict[str, set[str]] = {}
    name2bins: dict[str, list[str]] = {}
    for binary, decomp in bin2decomp.items():
        bin2names[binary] = {d.name for d in decomp}
        for d in decomp:
            accumulate(name2bins, d.name, binary)
    for bins in name2bins.values():
        bins.sort(key=lambda b: len(bin2names[b]))
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                smaller = bins[i]
                larger = bins[j]
                if smaller in bin2decomp and bin2names[smaller].issubset(bin2names[larger]):
                    del bin2decomp[smaller]
    del bin2names, name2bins

    call_graphs: dict[str, dict[str, list[str]]] = {}
    for binhash, decomp in bin2decomp.items():
        # Some functions in decomp may not be able to get canonical decompiled code or names. Those fields will be 
        # set to None. Leave them here for now to help the decomp/original matching process. They'll be filtered out later.
        call_graphs[binhash] = canonicalize_function_names(decomp)

    ### Process preprocessed source code.
    preprocessed_files = [Path(f).absolute() for f in scandir(preprocessed_location / repo) if f.is_file()]

    fnname2source: dict[str, list[str]] = {} # Index by function name
    preprocessed_functions: dict[str, list[PreprocessedFunction]] = {} # Index by function name. Values are deduplicated.
    fnbyfile: dict[tuple[str, str], PreprocessedFunction] = {} # Index by (file name, function name)
    for f in preprocessed_files:
        try:
            root = parse_file(f)
        except PermissionError:
            continue
        if root.type == "ERROR": # f could be another type of text file (like a generated configure script or something).
            # print(f"Could not parse {f}", file=sys.stderr)
            continue
        try:
            types = find_types(root)
        except NotImplementedError: # TODO: investigate why this occurs
            continue

        functions = find_functions(root, types)
        for fn in functions:
            accumulate(fnname2source, fn.name, f.name)
            fnbyfile[(f.name, fn.name)] = fn
            if fn.name in preprocessed_functions:
                if all(fn.text != other.text for other in preprocessed_functions[fn.name]):
                    preprocessed_functions[fn.name].append(fn)
            else:
                preprocessed_functions[fn.name] = [fn]

    # Index by binary
    matched: dict[str, list[MatchedFunction | None]] = {}

    ### Match decompiled functions with their corresponding definitions in the original function.
    for binary, decomps in bin2decomp.items():
        from_source_files: set[str] = set() # Of preprocessed hashes (not binary hashes)
        # First, identify the functions whose source provenance we're sure of: those that
        # have one unique definition in the source code.
        for decomp in decomps:
            if decomp.name in preprocessed_functions:
                matching_fns = preprocessed_functions[decomp.name]
                if len(matching_fns) == 1:
                    # Easy case. The name uniquely identifies the function.
                    accumulate(matched, binary, build_matched_function(decomp, matching_fns[0], repo))
                    from_source_files.update(fnname2source[decomp.name])
        # Next, attempt to use the files that we know other functions in the file came from to
        # choose the files that these functions came from.
        for decomp in decomps:
            if decomp.name in preprocessed_functions:
                matching_fns = preprocessed_functions[decomp.name]
                matching_files = fnname2source[decomp.name]
                if len(matching_fns) > 1 and len(fs := from_source_files.intersection(matching_files)) == 1:
                    accumulate(matched, binary, build_matched_function(decomp, fnbyfile[(fs.pop(), decomp.name)], repo))
    
    # The call graphs produced by canonicalize_function_names are unidirectional:
    # That is, they only contain information about outgoing calls, not incoming.
    # However, both callers and callees might be useful context for an idioms model.
    # Therefore, we add both to the call graph.
    bidirectional_call_graphs: dict[str, dict[str, list[str]]] = {}
    for binhash, unidirectional in call_graphs.items():
        # use insertion-order property of dictionaries to prevent duplicates while maintaining order.
        # this is done in the dictionaries associated with each function in 'bidirectional'
        bidirectional: dict[str, dict[str, None]] = {func: {} for func in unidirectional}
        for source, calls in unidirectional.items():
            for call in calls:
                if source != call:
                    if call in bidirectional: # may not be the case if it failed to parse. We don't have data for it, so ignore it.
                        bidirectional[source][call] = None # None is a placeholder; we only care about the keys.
                        bidirectional[call][source] = None
        bidirectional_call_graphs[binhash] = {func: list(calls) for func, calls in bidirectional.items()}

    # Build the MatchedBinaries now that all of the information necessary has been computed.
    output: list[MatchedBinary] = []
    for binhash, fns in matched.items():
        fns = list(filter(None, fns)) # remove those functions which have no canonical decompiled names or code. (These fail cause build_matched_function to return None.)
        matched_names = set(fn.name for fn in fns)
        unmatched: dict[str, str] = {decomp.name: decomp.canonical_code for decomp in bin2decomp[binhash] if decomp.name not in matched_names and decomp.canonical_code is not None}
        output.append(MatchedBinary(fns, binhash, repo, bidirectional_call_graphs[binhash], unmatched))

    return output

def _multiprocessing_prepare_repository(repoinfo: tuple[str, set[str]], 
                                        preprocessed_dir: Path, 
                                        decompiled_dir: Path
                                        ) -> list[MatchedBinary]:
    return prepare_repository(*repoinfo, preprocessed_dir, decompiled_dir)

##################################################
# Repository-level information 
##################################################

class BinaryMetadata(NamedTuple):
    binary_hash: str
    repository: str

def load_metadata(metadata_path: PathLike, include_hashes: set[str]) -> dict[str, list[BinaryMetadata]]:
    with open(metadata_path, "r") as fp:
        metadata = [
            BinaryMetadata(m['hash_name'], f"{m['repo_owner']}/{m['repo_name']}") 
            for m in json.load(fp)
            if m['hash_name'] in include_hashes
        ]
    

    repo2meta: dict[str, list[BinaryMetadata]] = {}
    for m in metadata:
        if m.repository in repo2meta:
            repo2meta[m.repository].append(m)
        else:
            repo2meta[m.repository] = [m]

    return repo2meta

def find_metadata(metadata_path: Path, include_hashes: set[str]) -> dict[str, list[BinaryMetadata]]:
    """Search `metadata_path` for binaries. The entries of `metadata_path` are assumed to be directories corresponding
    to repository owners and each owner directory should contain directories representing each repository. In turn,
    each repo should contin files that correspond to binaries.
    """
    repo2meta: dict[str, list[BinaryMetadata]] = {}
    for owner in tqdm(metadata_path.iterdir(), total=len(os.listdir(metadata_path))):
        for reponame in owner.iterdir():
            slug: str = owner.name + "/" + reponame.name
            meta: list[BinaryMetadata] = []
            for binary in reponame.iterdir():
                if binary.is_file() and binary.name in include_hashes:
                    meta.append(BinaryMetadata(
                        binary_hash=binary.name,
                        repository=slug
                    ))
            if len(meta) > 0:
                repo2meta[slug] = meta
    return repo2meta

def write_shard(filename: Path, contents: list[MatchedBinary]):
    with tarfile.open(filename, "w") as tf:
        for matchedbinary in contents:
            outbytes = bytes(json.dumps(matchedbinary.to_json(compact=True)), "utf8")
            # from https://bugs.python.org/issue22208
            info = tarfile.TarInfo(matchedbinary.binary_hash + ".json")
            info.size = len(outbytes)
            tf.addfile(info, fileobj=io.BytesIO(outbytes))
    

##################################################
# main function
##################################################

def main(args: argparse.Namespace):
    random.seed(args.random_seed)
    decompiled_dir = Path(args.decompiled_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    metadata_info_path = Path(args.metadata_info)
    repos_file = Path(args.deduplication_file_or_repo_list)
    output_dir = Path(args.output_dir)
    shard_size: int = args.shard_size
    dataset_size: Optional[int] = args.dataset_size
    valid_max_bins_per_repo = args.valid_max_bins_per_repo
    test_max_bins_per_repo = args.test_max_bins_per_repo
    single_split_name: str | None = args.single_split_name

    sys.setrecursionlimit(100000) # To handle very big functions

    if output_dir.exists():
        if not len(os.listdir(output_dir)) == 0:
            print(f"Output directory {output_dir} exists and is not empty.", file=sys.stderr)
            sys.exit(-1)
    else:
        os.mkdir(output_dir)
    
    # Check arguments and fail fast for obviously invalid arguments.
    for path in (decompiled_dir, preprocessed_dir, metadata_info_path):
        assert path.exists(), f"{path} does not exist."
    holdout_frac: float = args.holdout_set_size
    assert holdout_frac >= 0.0 and holdout_frac < 0.33, f"Holdout set size invalid: {holdout_frac} (must be in [0, 0.33))."
    assert shard_size > 0, f"Invalid shard size: {shard_size} (should be > 0)."
    assert repos_file.exists(), f"Deduplication file {repos_file} does not exist."

    # Find out for which binaries we have IDA-generated data
    # The original file name, which is the original hash of the binary is at [1]. For smaller binaries, this is the same,
    # but for bigger binaries it might be different.
    gzhashes: set[str] = {f.name.split("_")[0] for f in (decompiled_dir / "bins").iterdir()}
    print(f"Sample gzhashes: {list(itertools.islice(gzhashes, 5))}")

    # DIRTY's data generation code ignores repository information. Therefore, 
    # we track it in a separate file and map back to repositories based on binary hashes.
    if metadata_info_path.is_file():
        print("Loading metadata...")
        repometa = load_metadata(metadata_info_path, gzhashes)
    else:
        print("Finding metadata...")
        repometa = find_metadata(metadata_info_path, gzhashes)

    print(f"{len(repometa)} repos in metadata.")

    # The metadata is based on the decompiled code. Filter out repositories that we don't
    # have preprocessed code for.
    repometa = {
        repo: meta for repo, meta in repometa.items()
        if (preprocessed_dir / "repos" / repo).exists()
    }

    print(f"{len(repometa)} of theses repos exist at {preprocessed_dir}")

    # Filter out repositories that have no C files, and further deduplicate repositories
    if single_split_name is None:
        with open(repos_file, "r") as fp:
            dedup_info = json.load(fp)
        raw_duplicate_clusters: list[list[str]] = dedup_info["clusters"]
        no_c: set[str] = set(dedup_info["uncomputible"])

        repometa = {
            repo: meta for repo, meta in repometa.items()
            if repo not in no_c
        }

        print(f"{len(repometa)} of these repos have C code.")
    else:
        # the --single-split-name option assumes manual/prior deduplication
        with open(repos_file, "r") as fp:
            raw_duplicate_clusters = [[repo.strip()] for repo in fp.readlines()]
        assert all(len(c[0].split("/")) == 2 for c in raw_duplicate_clusters), f"In --single-split-name mode, expected one repo per line as the deduplication_file_or_repo_list argument."
        print(f"Read {len(raw_duplicate_clusters)} total repos")

    # Clusters may contain repositories that we weren't able to get preprocessed files for. Filter these out.
    duplicate_clusters = []
    for cluster in raw_duplicate_clusters:
        cluster = [c for c in cluster if c in repometa]
        if len(cluster) > 0:
            duplicate_clusters.append(cluster)

    # Split training/validation/test sets by repository. Binaries in a given
    # repository may share code, so splitting a repository across train and 
    # test sets may be unrealistically easy. We split by repository cluster because
    # we'll later select only one member from each cluster (the one with the most data).

    random.shuffle(duplicate_clusters)
    if dataset_size is not None:
        duplicate_clusters = duplicate_clusters[:dataset_size]
    if single_split_name is None:
        for cluster in duplicate_clusters:
            random.shuffle(cluster)
        holdout_size = int(holdout_frac * len(duplicate_clusters))
        testing_clusters = duplicate_clusters[:holdout_size]
        validation_clusters = duplicate_clusters[holdout_size:2*holdout_size]
        training_clusters = duplicate_clusters[2*holdout_size:]

    def repository_binaries(clusters: list[list[str]]) -> Iterator[tuple[str, set[str]]]:
        for cluster in clusters:
            for repo in cluster:
                yield repo, {m.binary_hash for m in repometa[repo]}

    def build_and_write_partition(clusters: list[list[str]], partition_name: str, max_bins_per_repo: Optional[int] = None) -> dict[str, list[str]]:
        cluster_sizes = {repo: len(cs) for cs in clusters for repo in cs}
        cluster_buffer: dict[str, list[list[MatchedBinary]]] = {}
        for cluster in clusters:
            cbuffer: list[list[MatchedBinary]] = []
            for repo in cluster:
                cluster_buffer[repo] = cbuffer
        repos = list(cluster_buffer)
        with multiprocessing.Pool(args.workers, maxtasksperchild=128) as pool:
            iterator: Iterator[list[MatchedBinary]] = pool.imap( # CANNOT change this to imap_unordered. Order is important in the zip'ed loop below
                iterable=repository_binaries(clusters),
                func=functools.partial(
                    _multiprocessing_prepare_repository, 
                    preprocessed_dir=preprocessed_dir / "repos", 
                    decompiled_dir=decompiled_dir / "bins"
                )
            )

            output_buffer: list[MatchedBinary] = []
            partition_bins: dict[str, list[str]] = {}
            shard_no = 0
            for repo, matched_binaries in tqdm(zip(repos, iterator), total=len(repos), desc=f"Processing {partition_name} set", dynamic_ncols=True):
                cluster_buffer[repo].append(matched_binaries)
                if len(cluster_buffer[repo]) == cluster_sizes[repo]:
                    # Select the repository from the cluster that contains the most functions.
                    matched_binaries = max(cluster_buffer[repo], key=lambda bins: sum(len(b.functions) for b in bins))
                    # Track the binaries for which we actually have original code definitions for at least one function in this repo.
                    partition_bins[repo] = [binary.binary_hash for binary in matched_binaries]
                    if max_bins_per_repo is None:
                        output_buffer.extend(matched_binaries)
                    else:
                        random.shuffle(matched_binaries)
                        output_buffer.extend(matched_binaries[:max_bins_per_repo])

                    while len(output_buffer) > shard_size:
                        to_write = output_buffer[:shard_size]
                        output_buffer = output_buffer[shard_size:]
                        write_shard(output_dir / f"{partition_name}-{shard_no}.tar", to_write)
                        shard_no += 1
                        del to_write
                del cluster_buffer[repo] # We don't want to be holding the whole dataset in memory. This helps get rid of the extra references to those items.
            
            write_shard(output_dir / f"{partition_name}-{shard_no}.tar", output_buffer)
            return partition_bins
        
    def write_repos(filename: Path, repos: Iterable[str]):
        with open(filename, "w") as fp:
            fp.write("\n".join(repos))
            fp.write("\n")
    
    if single_split_name is None:
        testing_bins = build_and_write_partition(testing_clusters, "test", test_max_bins_per_repo)
        validation_bins = build_and_write_partition(validation_clusters, "validation", valid_max_bins_per_repo)
        training_bins = build_and_write_partition(training_clusters, "train")

        make_bin_set = lambda bins: {b for repobins in bins.values() for b in repobins}
        flattened_test_bins = make_bin_set(testing_bins)
        flattened_validation_bins = make_bin_set(validation_bins)
        flattened_training_bins = make_bin_set(training_bins)

        print(f"Sizes of each set (# binaries): test: {len(flattened_test_bins)}, validation: {len(flattened_validation_bins)}, train: {len(flattened_training_bins)}")
        print(f"Test-train binary overlap {len(flattened_test_bins.intersection(flattened_training_bins))}")
        print(f"Validation-train overlap: {len(flattened_validation_bins.intersection(flattened_training_bins))}")
        print(f"Test-validation binary overlap: {len(flattened_test_bins.intersection(flattened_validation_bins))}")

        write_repos(output_dir / "test_repos.txt", testing_bins) # keys of the dictionary are the repos
        write_repos(output_dir / "validation_repos.txt", validation_bins)
        write_repos(output_dir / "train_repos.txt", training_bins)
    else:
        split_bins = build_and_write_partition(duplicate_clusters, single_split_name)
        write_repos(output_dir / f"{single_split_name}_repos.txt", split_bins)

    with open(output_dir / "command.txt", "w") as fp:
        fp.write(" ".join(sys.argv))
        fp.write("\n")


if __name__ == "__main__":
    main(get_args())
