import inspect
from graphgen.utils.log import CURRENT_LOGGER_VAR, set_logger


class BaseOperator(ABC):
    def __init__(self, working_dir: str = "cache", op_name: str = None):
        log_dir = os.path.join(working_dir, "logs")
        self.op_name = op_name or self.__class__.__name__

        try:
            ctx = ray.get_runtime_context()
            worker_id = ctx.get_actor_id() or ctx.get_worker_id()
            worker_id_short = worker_id[-6:] if worker_id else "driver"
        except Exception as e:
            print(
                "Warning: Could not get Ray worker ID, defaulting to 'local'. Exception:",
                e,
            )
            worker_id_short = "local"

        # e.g. cache/logs/ChunkService_a1b2c3.log
        log_file = os.path.join(log_dir, f"{self.op_name}_{worker_id_short}.log")

        self.logger = set_logger(
            log_file=log_file, name=f"{self.op_name}.{worker_id_short}",
            console_level=logging.ERROR, force=True
        )

        self.logger.info(
            "[%s] Operator initialized on Worker %s", self.op_name, worker_id_short
        )

    def __call__(
        self, batch: pd.DataFrame
    ) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
        logger_token = CURRENT_LOGGER_VAR.set(self.logger)
        try:
            result = self.process(batch)
            if inspect.isgenerator(result):
                yield from result
            else:
                yield result
        finally:
            CURRENT_LOGGER_VAR.reset(logger_token)

    @abstractmethod
    def process(self, batch):
        raise NotImplementedError("Subclasses must implement the process method.")

    def get_logger(self):
        return self.logger
