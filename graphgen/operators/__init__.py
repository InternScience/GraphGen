from .build_kg import BuildKGService
from .chunk import ChunkService
from .evaluate import EvaluateService
from .extract import ExtractService
from .generate import GenerateService
from .judge import JudgeService
from .partition import PartitionService
from .quiz import QuizService
from .read import read
from .rephrase import RephraseService
from .search import SearchService

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
}
