"""Every generator TEMPLATE_KEY must be loadable from the shipped example profile.

Regression for the #194 review nit: ``example.v1`` had no
``masked_fill_in_blank.json``, so applying the shipped example profile to
``MaskedFillInBlankGenerator`` raised ``prompt_profile_template_missing``.
"""

import pytest

from graphgen.models.generator import (
    AggregatedGenerator,
    AtomicGenerator,
    CoTGenerator,
    FillInBlankGenerator,
    MaskedFillInBlankGenerator,
    MultiAnswerGenerator,
    MultiChoiceGenerator,
    MultiHopGenerator,
    TrueFalseGenerator,
    VQAGenerator,
)
from graphgen.templates.prompt_profiles import load_profile_template

PROFILED_GENERATORS = [
    AggregatedGenerator,
    AtomicGenerator,
    CoTGenerator,
    FillInBlankGenerator,
    MaskedFillInBlankGenerator,
    MultiAnswerGenerator,
    MultiChoiceGenerator,
    MultiHopGenerator,
    TrueFalseGenerator,
    VQAGenerator,
]


@pytest.mark.parametrize("generator_cls", PROFILED_GENERATORS, ids=lambda cls: cls.__name__)
def test_example_v1_profile_covers_generator_template_key(generator_cls):
    """apply_prompt_profile("example.v1") must work for every shipped generator."""
    assert generator_cls.TEMPLATE_KEY, f"{generator_cls.__name__} must declare TEMPLATE_KEY"
    # load_profile_template validates the override structure against the
    # official template; a missing file raises prompt_profile_template_missing.
    assert load_profile_template("example.v1", generator_cls.TEMPLATE_KEY)
