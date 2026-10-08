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

__all__ = [
    "RECLOR_PARSER_VERSION",
    "RECLOR_VERIFIER_NAME",
    "RECLOR_VERIFIER_VERSION",
    "ParsedAnswer",
    "ReClorVerificationError",
    "parse_reclor_candidate",
    "verify_reclor_question",
]
