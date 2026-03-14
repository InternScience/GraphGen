from .build_kg import BuildKGService
from .chunk import ChunkService
from .evaluate import EvaluateService
from .extract import ExtractService
from .filter import FilterService
from .generate import GenerateService
from .judge import JudgeService
from .partition import PartitionService
from .quiz import QuizService
from .read import read
from .rephrase import RephraseService
from .search import SearchService
from .tree_pipeline import (
    BuildTreeKGService,
    HierarchyGenerateService,
    StructureAnalyzeService,
    TreeChunkService,
    TreeConstructService,
)

operators = {
    "read": read,
    "chunk": ChunkService,
    "build_kg": BuildKGService,
    "quiz": QuizService,
    "judge": JudgeService,
    "extract": ExtractService,
    "search": SearchService,
    "partition": PartitionService,
    "generate": GenerateService,
    "evaluate": EvaluateService,
    "rephrase": RephraseService,
    "filter": FilterService,
    "structure_analyze": StructureAnalyzeService,
    "hierarchy_generate": HierarchyGenerateService,
    "tree_construct": TreeConstructService,
    "tree_chunk": TreeChunkService,
    "build_tree_kg": BuildTreeKGService,
}
