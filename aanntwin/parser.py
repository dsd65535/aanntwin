"""This module contains convenience argument parsing functions"""
import argparse
import logging
from dataclasses import _MISSING_TYPE
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Tuple
from typing import Union


def _extract_type_and_nargs(candidate: Any) -> Tuple[type, Optional[str]]:
    """Convert a dataclass Field.type into a type and nargs for argparse"""

    if isinstance(candidate, type):
        return candidate, None

    origin = get_origin(candidate)
    if origin == Union:
        args = get_args(candidate)
        if len(args) != 2:
            raise ValueError(f"Candidate {candidate} is a Union of {len(args)} items")
        if args[0] is type(None):
            candidate, nargs = _extract_type_and_nargs(args[1])
        elif args[1] is type(None):
            candidate, nargs = _extract_type_and_nargs(args[0])
        else:
            raise ValueError(f"Candidate {candidate} is a bad Union")
        if nargs is None:
            return candidate, "?"
        if nargs == "+":
            return candidate, "*"
        raise RuntimeError

    if origin is list:
        args = get_args(candidate)
        if len(args) != 1:
            raise ValueError(f"Candidate {candidate} is a List of {len(args)} items")
        candidate, nargs = _extract_type_and_nargs(args[0])
        if nargs is None:
            return candidate, "+"
        raise RuntimeError

    raise ValueError(
        f"Candidate {candidate} is not a type and has an unsupported origin {origin}"
    )


def add_arguments_from_dataclass_fields(
    parent: type,
    parser: argparse.ArgumentParser,
    help_dict: Optional[Dict[str, str]] = None,
) -> None:
    """Add an argument to a parser for each member of a dataclass"""

    if not is_dataclass(parent):
        raise ValueError(f"Parent {parent} is not a dataclass")

    for field in parent.__dataclass_fields__.values():
        if not field.init:
            continue

        name = field.name
        kwargs: Dict[str, Any] = {}

        try:
            kwargs["type"], nargs = _extract_type_and_nargs(field.type)
        except ValueError:
            logging.exception(
                "Dataclass field %s has unrecognized type %s", name, field.type
            )
            continue

        if nargs is not None:
            kwargs["nargs"] = nargs

        default = field.default
        if not isinstance(default, _MISSING_TYPE):
            kwargs["default"] = default
            if kwargs["type"] is bool:
                if default is True:
                    name = f"no_{name}"

        if help_dict is not None and name in help_dict:
            kwargs["help"] = help_dict[name]

        parser.add_argument(f"--{name}", **kwargs)
