from .build_kg import BuildKGService
from .chunk import ChunkService
from .extract import extract
from .generate import generate_qas
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
    "extract_info": extract,
    "search_all": search_all,
    "partition": PartitionService,
    "generate_qas": generate_qas,
}
