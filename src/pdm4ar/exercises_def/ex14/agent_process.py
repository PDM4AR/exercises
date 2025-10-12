import multiprocessing as mp
import multiprocessing.connection
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

import cloudpickle as cp
from dg_commons.sim.agents import Agent
from dg_commons.sim.simulator_structures import InitSimObservations, SimObservations


@dataclass
class _Msg:
    id: str
    op: Literal["call", "getattr", "setattr", "close", "ok", "err"]
    name: Optional[str] = None
    payload: Optional[bytes] = None
    error: Optional[str] = None
    wall_time: Optional[float] = None


def _dumps(obj: Any) -> bytes:
    return cp.dumps(obj)


def _loads(payload: bytes) -> Any:
    return cp.loads(payload)


def _worker_loop(conn: multiprocessing.connection.Connection, ctor_bytes: bytes, init_bytes: bytes) -> None:
    """Worker process: construct agent, then serve RPC."""
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

            elif msg.op == "getattr":
                val = getattr(agent, msg.name)
                if callable(val):
                    raise TypeError(f"{msg.name} is callable; use 'call'")
                conn.send(_Msg(id=msg.id, op="ok", payload=_dumps(val)))

            elif msg.op == "setattr":
                val = _loads(msg.payload)
                setattr(agent, msg.name, val)
                conn.send(_Msg(id=msg.id, op="ok"))

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
    """Proxy for remote method calls (transparent timing optional)."""

    def __init__(self, owner: "AgentProcess", name: str):
        self._owner = owner
        self._name = name

    def __call__(self, *args, timeout: float | None = None, **kwargs):
        return self._owner._rpc_call("call", self._name, (args, kwargs), timeout=timeout)


class AgentProcess(Agent):
    """
    Linux-optimized process-isolated agent wrapper with wall-time measurement.
    """

    def __init__(self, agent_ctor, *init_args, **init_kwargs):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = mp.Process(
            target=_worker_loop,
            args=(child_conn, _dumps(agent_ctor), _dumps((init_args, init_kwargs))),
            daemon=True,
        )
        self._proc.start()
        msg = self._conn.recv()
        if msg.op != "ok":
            raise RuntimeError(f"Agent failed to start:\n{msg.error}")
        self._closed = False
        self._last_function_call_time = 0.0

    # --- Agent interface ---
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self._rpc_call("call", "on_episode_init", (init_sim_obs,))

    def get_commands(self, sim_obs: SimObservations) -> Any:
        return self._rpc_call("call", "get_commands", (sim_obs,))

    # --- Transparent attribute access ---
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError
        try:
            return self._rpc_call("getattr", name, None)
        except TypeError:
            return _MethodProxy(self, name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            return super().__setattr__(name, value)
        self._rpc_call("setattr", name, value)

    def close(self, timeout: float | None = 2.0):
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _rpc_call(
        self, op: str, name: Optional[str], body: Any, timeout: Optional[float] = None, return_time: bool = False
    ):
        """Perform an RPC and optionally return (result, wall_time)."""
        msg_id = uuid.uuid4().hex
        payload = None if body is None else _dumps(body)
        self._conn.send(_Msg(id=msg_id, op=op, name=name, payload=payload))
        if timeout is not None and not self._conn.poll(timeout):
            raise TimeoutError(f"Timeout on RPC {op}({name})")
        reply: _Msg = self._conn.recv()
        if reply.op == "ok":
            result = None if reply.payload is None else _loads(reply.payload)
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
