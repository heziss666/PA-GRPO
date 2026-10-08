"""Strict candidate verification contracts for controlled data production."""

from .reclor import (
    RECLOR_PARSER_VERSION,
    RECLOR_VERIFIER_NAME,
    RECLOR_VERIFIER_VERSION,
    ParsedAnswer,
    ReClorVerificationError,
    parse_reclor_candidate,
    verify_reclor_question,
)
from .math import (
    MATH_CANONICAL_REPRESENTATION_VERSION,
    MATH_PARSER_VERSION,
    MATH_VERIFIER_NAME,
    MATH_VERIFY_VERSION,
    BoxedAnswer,
    DependencyContractError,
    MathVerificationError,
    extract_math_candidate,
    extract_math_gold,
    verify_math_question,
)

__all__ = [
    "RECLOR_PARSER_VERSION",
    "RECLOR_VERIFIER_NAME",
    "RECLOR_VERIFIER_VERSION",
    "MATH_CANONICAL_REPRESENTATION_VERSION",
    "MATH_PARSER_VERSION",
    "MATH_VERIFIER_NAME",
    "MATH_VERIFY_VERSION",
    "BoxedAnswer",
    "DependencyContractError",
    "MathVerificationError",
    "ParsedAnswer",
    "ReClorVerificationError",
    "parse_reclor_candidate",
    "extract_math_candidate",
    "extract_math_gold",
    "verify_math_question",
    "verify_reclor_question",
]
