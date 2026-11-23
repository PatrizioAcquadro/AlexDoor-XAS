"""Bounded stepping and signal handling for simulated collection."""

import signal
from time import perf_counter


class CollectionBudgetExceeded(TimeoutError):
    pass


def collection_termination_reason(error):
    if isinstance(error, CollectionBudgetExceeded):
        return "budget_termination"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    return "collection_exception"


def install_interruption(env, *, max_wall_s=None, max_sim_s=None):
    """Raise outside Kit callbacks, then restore unguarded stepping for the stop."""
    started = perf_counter()
    pending = []
    env.collection_physics_s = 0.0
    original_step = env.step

    def request(signum, _frame):
        import faulthandler

        faulthandler.dump_traceback(all_threads=True)
        pending.append(signum)

    def check():
        if max_wall_s is not None and perf_counter() - started >= max_wall_s:
            raise CollectionBudgetExceeded("collection_wall_budget")
        if max_sim_s is not None and env.collection_physics_s >= max_sim_s:
            raise CollectionBudgetExceeded("collection_simulation_budget")
        if pending and pending[0] == signal.SIGUSR1:
            raise CollectionBudgetExceeded("collection_resource_budget")
        if pending:
            raise KeyboardInterrupt(f"Collection interrupted by signal {pending[0]}")

    def step(*args, **kwargs):
        check()
        result = original_step(*args, **kwargs)
        env.collection_physics_s += getattr(env, "step_dt", 0.0)
        check()
        return result

    env.step = step
    env.collection_check = check
    handlers = {sig: signal.signal(sig, request) for sig in (signal.SIGTERM, signal.SIGUSR1)}

    def restore():
        env.step = original_step
        env.collection_check = None
        for sig, previous in handlers.items():
            signal.signal(sig, previous)

    return restore
