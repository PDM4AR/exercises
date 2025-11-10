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

    def __setattr__(self, name: str, value):
        """Forward non-private attribute sets to the remote agent.

        Attributes starting with '_' are treated as local and set on the
        proxy object. Other attributes are sent to the remote agent by
        calling its __setattr__ via RPC so the remote instance stays in
        sync (e.g. setting `player.name = ...`). We also mirror the
        attribute locally for convenience.
        """
        # Local internals: set directly
        if name.startswith("_"):
            return object.__setattr__(self, name, value)

        # If the connection or rpc call machinery isn't ready yet, fall
        # back to setting locally.
        if not hasattr(self, "_conn") or getattr(self, "_closed", True):
            return object.__setattr__(self, name, value)

        # Forward to remote agent: call its __setattr__
        try:
            # Use the existing RPC mechanism to call remote __setattr__
            self._rpc_call("call", "__setattr__", ((name, value), {}))
            # Mirror locally so subsequent local reads see the same value
            object.__setattr__(self, name, value)
        except Exception:
            # If remote set fails for any reason, fall back to local set
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        """Attempt to retrieve non-local attributes from the remote agent.

        Called only when attribute lookup on the proxy fails. We forward
        the lookup to the remote agent's __getattribute__ via RPC. If the
        remote lookup fails, raise AttributeError.
        """
        # Do not proxy private attributes
        if name.startswith("_"):
            raise AttributeError(name)

        # If proxy isn't fully initialized, behave like normal attribute error
        if not hasattr(self, "_conn") or getattr(self, "_closed", True):
            raise AttributeError(name)

        try:
            return self._rpc_call("call", "__getattribute__", ((name,), {}))
        except Exception as e:
            raise AttributeError(f"Remote attribute error: {name}: {e}")

    # --- Agent interface ---
    def get_commands(self, sim_obs: SimObservations) -> Any:
        return _MethodProxy(self, "get_commands")(sim_obs)

    def on_get_extra(self) -> Optional[Any]:
        return _MethodProxy(self, "on_get_extra")()

    # --- state tracking interface ---
    def set_capacity(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if self._current_load > capacity:
            raise ValueError(f"Cannot set capacity to {capacity}; current load {self._current_load} would exceed it")
        self._capacity = capacity

    def get_capacity(self) -> int:
        return self._capacity

    def set_current_load(self, load: int):
        if not isinstance(load, int):
            raise TypeError("Load must be an integer")
        if load < 0:
            raise ValueError("Load must be a non-negative integer")
        if load > self._capacity:
            raise ValueError(f"Agent load {load} exceeds capacity {self._capacity}")
        self._current_load = load

    def get_current_load(self) -> int:
        return self._current_load

    def grab_goal(self):
        if self._current_load + 1 > self._capacity:
            raise ValueError(f"Agent load {self._current_load + 1} exceeds capacity {self._capacity}")
        self._current_load += 1

    def deliver_goal(self):
        if self._current_load - 1 < 0:
            raise ValueError(f"Agent load {self._current_load - 1} is negative")
        self._current_load -= 1

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
