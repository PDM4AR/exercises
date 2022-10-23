from typing import Union, Any
import multiprocessing
import traceback
from functools import wraps
from pdm4ar.exercises_def import logger
from pdm4ar.exercises_def.structures_memory import set_memory_limit

class TestCaseTimeoutException(Exception):
    pass


def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    ########
    set_memory_limit()  # Limited to 8GB on the evaluation server
    ########
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        logger.warn(f"Failed because of:\n {e.args} \n{''.join(traceback.format_tb(e.__traceback__))}")
        return
    send_end.send(result)


def run_with_timer(func, max_execution_time) -> Union[Any, Exception]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func

        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        if recv_end.poll(max_execution_time):
            result = recv_end.recv()
        else:
            result = TestCaseTimeoutException("Exceeded test case timeout.")
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            result = TestCaseTimeoutException("Exceeded test case timeout.")

        return result

    return wrapper
