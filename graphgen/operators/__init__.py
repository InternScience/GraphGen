from .build_kg import BuildKGService
from .chunk import ChunkService
from .extract import ExtractService
from .generate import GenerateService
from .judge import JudgeService
from .partition import PartitionService
from .quiz import QuizService
from .read import read
from .search import search_all

operators = {
    "read": read,
    "chunk": ChunkService,
    "build_kg": BuildKGService,
    "quiz": QuizService,
    "judge": JudgeService,
    "extract": ExtractService,
    "search": search_all,
    "partition": PartitionService,
    "generate": GenerateService,
}
