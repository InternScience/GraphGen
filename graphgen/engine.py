"""
orchestration engine for GraphGen
"""

import threading
from typing import Any, Callable, List


class Context(dict):
    _lock = threading.Lock()

    def set(self, k, v):
        with self._lock:
            self[k] = v

    def get(self, k, default=None):
        with self._lock:
            return super().get(k, default)


class OpNode:
    def __init__(
        self, name: str, deps: List[str], func: Callable[["OpNode", Context], Any]
    ):
        self.name, self.deps, self.func = name, deps, func


def op(name: str, deps: List[str] = None):
    def decorator(f: Callable[["OpNode", Context], Any]):
        return OpNode(name, deps or [], f)

    return decorator


class Engine:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run(self, ops: List[OpNode], ctx: Context):
        name2op = {operation.name: operation for operation in ops}

        # topological sort
        graph = {n: set(name2op[n].deps) for n in name2op}
        topo = []
        q = [n for n, d in graph.items() if not d]
        while q:
            cur = q.pop(0)
            topo.append(cur)
            for child in [c for c, d in graph.items() if cur in d]:
                graph[child].remove(cur)
                if not graph[child]:
                    q.append(child)

        if len(topo) != len(ops):
            raise ValueError(
                "Cyclic dependencies detected among operations."
                "Please check your configuration."
            )

        # semaphore for max_workers
        sem = threading.Semaphore(self.max_workers)
        done = {n: threading.Event() for n in name2op}
        exc = {}

        def _exec(n: str):
            with sem:
                for d in name2op[n].deps:
                    done[d].wait()
                if any(d in exc for d in name2op[n].deps):
                    exc[n] = Exception("Skipped due to failed dependencies")
                    done[n].set()
                    return
                try:
                    name2op[n].func(name2op[n], ctx)
                except Exception as e:  # pylint: disable=broad-except
                    exc[n] = e
                done[n].set()

        ts = [threading.Thread(target=_exec, args=(n,), daemon=True) for n in topo]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if exc:
            raise RuntimeError(f"Some operations failed: {exc}")
