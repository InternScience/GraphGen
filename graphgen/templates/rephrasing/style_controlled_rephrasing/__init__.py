from .critical_analysis_rephrasing import CRITICAL_ANALYSIS_REPHRASING_PROMPTS
from .cross_domain_analogy_rephrasing import CROSS_DOMAIN_ANALOGY_REPHRASING_PROMPTS
from .executive_summary_rephrasing import EXECUTIVE_SUMMARY_REPHRASING_PROMPTS
from .first_person_narrative_rephrasing import FIRST_PERSON_NARRATIVE_REPHRASING_PROMPTS
from .historical_evolution_perspective_rephrasing import (
    HISTORICAL_EVOLUTION_PERSPECTIVE_REPHRASING_PROMPTS,
)
from .popular_science_rephrasing import POPULAR_SCIENCE_REPHRASING_PROMPTS
from .qa_dialogue_format_rephrasing import QA_DIALOGUE_FORMAT_REPHRASING_PROMPTS
from .technical_deep_dive_rephrasing import TECHNICAL_DEEP_DIVE_REPHRASING_PROMPTS

STYLE_CONTROLLED_REPHRASING_PROMPTS = {
    "popular_science": POPULAR_SCIENCE_REPHRASING_PROMPTS,
    "critical_analysis": CRITICAL_ANALYSIS_REPHRASING_PROMPTS,
    "cross_domain_analogy": CROSS_DOMAIN_ANALOGY_REPHRASING_PROMPTS,
    "technical_deep_dive": TECHNICAL_DEEP_DIVE_REPHRASING_PROMPTS,
    "executive_summary": EXECUTIVE_SUMMARY_REPHRASING_PROMPTS,
    "first_person_narrative": FIRST_PERSON_NARRATIVE_REPHRASING_PROMPTS,
    "historical_evolution_perspective": HISTORICAL_EVOLUTION_PERSPECTIVE_REPHRASING_PROMPTS,
    "qa_dialogue_format": QA_DIALOGUE_FORMAT_REPHRASING_PROMPTS,
}
