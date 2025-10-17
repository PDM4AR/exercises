import multiprocessing as mp
import multiprocessing.connection
import time
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing.util import Finalize
from typing import Any, Literal, Optional

import cloudpickle as cp
from dg_commons.sim.agents import Agent
from dg_commons.sim.simulator_structures import InitSimObservations, SimObservations
from pdm4ar.exercises_def.ex14.restricted_loads import restricted_loads

MsgOp = Literal["call", "close", "ok", "err"]


@dataclass
class _Msg:
    id: str
    op: MsgOp
    name: Optional[str] = None
    payload: Optional[bytes] = None
    error: Optional[str] = None
    wall_time: Optional[float] = None


def _dumps(obj: Any) -> bytes:
    return cp.dumps(obj)


def _loads(payload: bytes) -> Any:
    return cp.loads(payload)


def _restricted_loads(payload: bytes) -> Any:
    allowed_modules = ["dg_commons", "numpy"]
    return restricted_loads(payload, allowed_modules=allowed_modules)


def _worker_loop(conn: multiprocessing.connection.Connection, ctor_bytes: bytes, init_bytes: bytes) -> None:
    """Worker process: construct agent, then serve RPC."""
    with conn:
        try:
            ctor = _loads(ctor_bytes)
            args, kwargs = _loads(init_bytes)
            agent = ctor(*args, **kwargs)
        except Exception:
            conn.send(_Msg(id="init", op="err", error=traceback.format_exc()))
            return
        else:
            conn.send(_Msg(id="init", op="ok"))

        while True:
            msg: _Msg = conn.recv()
            try:
                if msg.op == "close":
                    conn.send(_Msg(id=msg.id, op="ok"))
                    break

                elif msg.op == "call":
                    args, kwargs = _loads(msg.payload)
                    t0 = time.perf_counter()
                    result = getattr(agent, msg.name)(*args, **kwargs)
                    wall = time.perf_counter() - t0
                    conn.send(_Msg(id=msg.id, op="ok", payload=_dumps(result), wall_time=wall))

                else:
                    raise ValueError(f"Unknown op {msg.op}")

            except Exception:
                conn.send(_Msg(id=msg.id, op="err", error=traceback.format_exc()))


class _MethodProxy:
    """Proxy for remote method calls."""

    def __init__(self, owner: "AgentProcess", name: str):
        self._owner = owner
        self._name = name

    def __call__(self, *args, **kwargs):
        return self._owner._rpc_call("call", self._name, (args, kwargs))


class AgentProcess(Agent):
    """
    Linux-optimized process-isolated agent wrapper with wall-time measurement.
    """

    def __init__(self, agent_ctor, *init_args, **init_kwargs):
        if init_args is None:
            init_args = ()
        if init_kwargs is None:
            init_kwargs = {}

        parent_conn, child_conn = mp.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = mp.Process(
            target=_worker_loop,
            args=(child_conn, _dumps(agent_ctor), _dumps((init_args, init_kwargs))),
        )
        self._proc.start()
        child_conn.close()

        self._finalizer = Finalize(self, self.close, exitpriority=10)

        msg = self._conn.recv()
        if msg.op != "ok":
            raise RuntimeError(f"Agent failed to start:\n{msg.error}")
        self._closed = False
        self._last_function_call_time = 0.0

        # --- state tracking ---
        self._capacity = 1  # default capacity
        self._current_load = 0

    # --- Agent interface ---
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        return _MethodProxy(self, "on_episode_init")(init_sim_obs)

    def get_commands(self, sim_obs: SimObservations) -> Any:
        return _MethodProxy(self, "get_commands")(sim_obs)

    def on_get_extra(self) -> Optional[Any]:
        return _MethodProxy(self, "on_get_extra")()

    # --- state tracking interface ---
    def set_capacity(self, capacity: int):
        assert capacity > 0 and isinstance(capacity, int), "Capacity must be a positive integer"
        self._capacity = capacity

    def get_capacity(self) -> int:
        return self._capacity

    def set_current_load(self, load: int):
        assert 0 <= load <= self._capacity and isinstance(
            load, int
        ), "Load must be a non-negative integer within capacity"
        self._current_load = load

    def get_current_load(self) -> int:
        return self._current_load

    # --- internal ---
    def close(self, timeout: Optional[float] = 5.0):
        if self._closed:
            return
        try:
            self._rpc_call("close", None, None, timeout=timeout)
        finally:
            if self._proc.is_alive():
                self._proc.join(timeout)
                if self._proc.is_alive():
                    self._proc.kill()
            self._closed = True
            self._conn.close()

    def __del__(self):
        self.close()

    def _rpc_call(
        self,
        op: MsgOp,
        name: Optional[str],
        body: Optional[tuple[tuple, dict]],
        timeout: Optional[float] = None,
        return_time: bool = False,
    ):
        """Perform an RPC and optionally return (result, wall_time)."""
        msg_id = uuid.uuid4().hex
        payload = None if body is None else _dumps(body)
        self._conn.send(_Msg(id=msg_id, op=op, name=name, payload=payload))
        if timeout is not None and not self._conn.poll(timeout):
            raise TimeoutError(f"Timeout on RPC {op}({name})")
        reply: _Msg = self._conn.recv()
        if reply.op == "ok":
            result = None if reply.payload is None else _restricted_loads(reply.payload)
            self._last_function_call_time = reply.wall_time
            if return_time:
                return result, reply.wall_time
            return result
        raise RuntimeError(f"Remote error on {op}({name}):\n{reply.error}")

    # --- public utility ---
    def call_timed(self, name: str, *args, timeout=None, **kwargs):
        """
        Like getattr(self, name)(...), but returns (result, wall_time).
        Example:
            action, t = agent.call_timed("act", obs)
        """
        return self._rpc_call("call", name, (args, kwargs), timeout=timeout, return_time=True)
