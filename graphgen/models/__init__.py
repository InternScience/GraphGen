from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import (
        AccuracyEvaluator,
        LengthEvaluator,
        MTLDEvaluator,
        RewardEvaluator,
        StructureEvaluator,
        UniEvaluator,
    )
    from .filter import RangeFilter
    from .generator import (
        AggregatedGenerator,
        AtomicGenerator,
        CoTGenerator,
        FillInBlankGenerator,
        MultiAnswerGenerator,
        MultiChoiceGenerator,
        MultiHopGenerator,
        QuizGenerator,
        TrueFalseGenerator,
        VQAGenerator,
        MaskedFillInBlankGenerator,
    )
    from .kg_builder import LightRAGKGBuilder, MMKGBuilder
    from .llm import HTTPClient, OllamaClient, OpenAIClient
    from .partitioner import (
        AnchorBFSPartitioner,
        BFSPartitioner,
        DFSPartitioner,
        ECEPartitioner,
        LeidenPartitioner,
        TriplePartitioner,
    )
    from .reader import (
        CSVReader,
        HuggingFaceReader,
        JSONReader,
        ParquetReader,
        PDFReader,
        PickleReader,
        RDFReader,
        TXTReader,
    )
    from .rephraser import StyleControlledRephraser
    from .searcher.db.interpro_searcher import InterProSearch
    from .searcher.db.ncbi_searcher import NCBISearch
    from .searcher.db.rnacentral_searcher import RNACentralSearch
    from .searcher.db.uniprot_searcher import UniProtSearch
    from .searcher.kg.wiki_search import WikiSearch
    from .searcher.web.bing_search import BingSearch
    from .searcher.web.google_search import GoogleSearch
    from .splitter import ChineseRecursiveTextSplitter, RecursiveCharacterSplitter
    from .tokenizer import Tokenizer

_import_map = {
    # Evaluator
    "AccuracyEvaluator": ".evaluator",
    "LengthEvaluator": ".evaluator",
    "MTLDEvaluator": ".evaluator",
    "RewardEvaluator": ".evaluator",
    "StructureEvaluator": ".evaluator",
    "UniEvaluator": ".evaluator",
    # Filter
    "RangeFilter": ".filter",
    # Generator
    "AggregatedGenerator": ".generator",
    "AtomicGenerator": ".generator",
    "CoTGenerator": ".generator",
    "FillInBlankGenerator": ".generator",
    "MultiAnswerGenerator": ".generator",
    "MultiChoiceGenerator": ".generator",
    "MultiHopGenerator": ".generator",
    "QuizGenerator": ".generator",
    "TrueFalseGenerator": ".generator",
    "VQAGenerator": ".generator",
    "MaskedFillInBlankGenerator": ".generator",
    # KG Builder
    "LightRAGKGBuilder": ".kg_builder",
    "MMKGBuilder": ".kg_builder",
    # LLM
    "HTTPClient": ".llm",
    "OllamaClient": ".llm",
    "OpenAIClient": ".llm",
    # Partitioner
    "AnchorBFSPartitioner": ".partitioner",
    "BFSPartitioner": ".partitioner",
    "DFSPartitioner": ".partitioner",
    "ECEPartitioner": ".partitioner",
    "LeidenPartitioner": ".partitioner",
    "TriplePartitioner": ".partitioner",
    # Reader
    "CSVReader": ".reader",
    "JSONReader": ".reader",
    "ParquetReader": ".reader",
    "PDFReader": ".reader",
    "PickleReader": ".reader",
    "RDFReader": ".reader",
    "TXTReader": ".reader",
    "HuggingFaceReader": ".reader",
    # Searcher
    "InterProSearch": ".searcher.db.interpro_searcher",
    "NCBISearch": ".searcher.db.ncbi_searcher",
    "RNACentralSearch": ".searcher.db.rnacentral_searcher",
    "UniProtSearch": ".searcher.db.uniprot_searcher",
    "WikiSearch": ".searcher.kg.wiki_search",
    "BingSearch": ".searcher.web.bing_search",
    "GoogleSearch": ".searcher.web.google_search",
    # Splitter
    "ChineseRecursiveTextSplitter": ".splitter",
    "RecursiveCharacterSplitter": ".splitter",
    # Tokenizer
    "Tokenizer": ".tokenizer",
    # Rephraser
    "StyleControlledRephraser": ".rephraser",
}


def __getattr__(name):
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_import_map.keys())
