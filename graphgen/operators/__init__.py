from .build_kg import BuildKGService
from .chunk import ChunkService
from .extract import ExtractService
from .generate import GenerateService
from .judge import JudgeService
from .partition import PartitionService
from .quiz import QuizService
from .read import read
from .search import SearchService
from .evaluate import EvaluateService


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
}
