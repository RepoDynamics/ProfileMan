"""Function profiler."""

import timeit
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

InputArgs = Sequence[Any]
InputKwargs = dict[str, Any]
Output = Any
ArgSize = int
ArgGen = Callable[[ArgSize], tuple[InputArgs, InputKwargs]]


class FunctionProfiler:
    """Profile (and compare) one or several functions.

    Parameters
    ----------
    funcs
        Function(s) to be profiled.
    arg_gens
        Argument generators for each function in `funcs`.
        Each generator receives a sequence of numbers
        indicating different input sizes
        for the respective function's arguments,
        and yields input arguments for that function in form of a tuple.
        That is, each function `f` in `funcs` should be callable
        with the corresponding generator `g` in `arg_gens`,
        for example as `[f(*args) for args in g([1, 10, 100])]`.
    """

    def __init__(
        self,
        function: Callable | Sequence[Callable],
        input_gen: ArgGen | Sequence[ArgGen],
        function_name: Sequence[str] | None = None,
        output_evaluator: Callable[[ArgSize, Sequence[InputArgs], Sequence[InputKwargs], Sequence[Output]], None] | None = None,
    ):
        self._funcs: list[Callable] = function if isinstance(function, Sequence) else [function]
        self._input_gens: list[ArgGen] = input_gen if isinstance(input_gen, Sequence) else [input_gen]
        if len(self._funcs) != len(self._input_gens):
            raise ValueError(
                "Parameters `funcs` and `arg_gens` expect inputs with identical lengths, "
                f"but `funcs` had a length of {len(self._funcs)}, "
                f"while `arg_gens` had  {len(self._input_gens)}."
            )
        if function_name is None:
            function_name = []
            for func_idx, func in enumerate(self._funcs):
                func_name = f"\u200b{func.__module__}.{func.__qualname__}"
                if func_name in function_name:
                    func_name = f"{func_name} ({func_idx})"
                function_name.append(func_name)
        elif len(function_name) != len(self._funcs):
            raise ValueError(
                "Parameter `func_names` must have the same length as `funcs`, "
                f"but it had a length of {len(function_name)}, while `funcs` had {len(self._funcs)}."
            )
        self._func_names: list[str] = function_name
        self._output_eval = output_evaluator

        self._arg_sizes: list[int] = None
        self._runs: int = None
        self._loops_per_run: int = None
        self._results: np.ndarray = None
        return

    def profile(
        self,
        arg_sizes: Sequence[int],
        runs: int = 100,
        loops_per_run: int = 1,
    ) -> None:
        """Profile all functions.

        Parameters
        ----------
        arg_sizes
            Different argument sizes to profile the functions with.
        runs
            Number of times the profiling is repeated for each argument size.
            Higher values provide more accurate results.
            The shortest duration between all runs will be selected,
            as it represents the most accurate duration.
        loops_per_run
            Number of times each run is repeated.
            For each run, the average of all loops will be selected.
        """
        self._arg_sizes = arg_sizes
        self._runs = runs
        self._loops_per_run = loops_per_run
        results = []
        for func, arg_gen in zip(self._funcs, self._input_gens, strict=True):
            shortest_runtime_per_argsize_for_func: list[list[float]] = []
            for args, kwargs in arg_gen(arg_sizes):
                all_loop_times = timeit.repeat(
                    lambda: func(*args, **kwargs),
                    repeat=self._runs,
                    number=self._loops_per_run
                )
                shortest_loop_time = np.min(all_loop_times)
                shortest_run_time = shortest_loop_time / self._loops_per_run
                shortest_runtime_per_argsize_for_func.append(shortest_run_time)
            results.append(shortest_runtime_per_argsize_for_func)
        self._results = np.array(results)
        return

    def profile(
        self,
        arg_sizes: Sequence[int],
        runs: int = 100,
        loops_per_run: int = 1,
    ) -> None:
        """Profile all functions.

        Parameters
        ----------
        arg_sizes
            Different argument sizes to profile the functions with.
        runs
            Number of times the profiling is repeated for each argument size.
            Higher values provide more accurate results.
            The shortest duration between all runs will be selected,
            as it represents the most accurate duration.
        loops_per_run
            Number of times each run is repeated.
            For each run, the average of all loops will be selected.
        """
        self._arg_sizes = arg_sizes
        self._runs = runs
        self._loops_per_run = loops_per_run
        self._results = np.empty((len(self._funcs), len(arg_sizes)), dtype=float)
        for arg_size_idx, arg_size in enumerate(arg_sizes):
            input_args = []
            input_kwargs = []
            outputs = []
            for func_idx, (func, arg_gen) in enumerate(zip(self._funcs, self._input_gens, strict=True)):
                args, kwargs = arg_gen(arg_size)
                if self._output_eval is not None:
                    input_args.append(args)
                    input_kwargs.append(kwargs)
                    outputs.append(func(*args, **kwargs))
                all_loop_times = timeit.repeat(
                    lambda: func(*args, **kwargs),
                    repeat=self._runs,
                    number=self._loops_per_run
                )
                shortest_loop_time = np.min(all_loop_times)
                shortest_run_time = shortest_loop_time / self._loops_per_run
                self._results[func_idx, arg_size_idx] = shortest_run_time
            if self._output_eval is not None:
                self._output_eval(input_args, input_kwargs, outputs)
        return self._results

    def plot(self, show: bool = True):
        if self._results is None:
            raise ValueError(
                "No profiling has been performed yet; call `FunctionProfiler.profile` first."
            )
        fig, ax = plt.subplots()
        artists = []
        for func_name, result in zip(self._func_names, self._results, strict=True):
            (line,) = ax.plot(
                self._arg_sizes,
                result,
                marker=".",
                label=func_name,
            )
            artists.append(line)
        ax.legend(handles=artists, loc="best")
        plt.xlabel("Input size")
        plt.ylabel("Time [s]")
        plt.xscale("log")
        plt.yscale("log")
        if show:
            plt.show()
        return fig, ax
