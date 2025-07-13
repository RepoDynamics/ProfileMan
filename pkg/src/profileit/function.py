"""Function profiler."""

import timeit
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

InputArgs = Sequence[Any]
InputKwargs = dict[str, Any]
Output = Any
InputSize = int
ArgGen = Callable[[InputSize], tuple[InputArgs, InputKwargs]]


class FunctionProfiler:
    """Profile (and compare) one or several functions.

    Parameters
    ----------
    function
        Function(s) to be profiled.
    input_gen
        Input argument generator for each function in `function`.
        Each generator must accept an integer as input,
        indicating the size of the input(s) to generate.
        The generator should return a tuple of two elements:
        - A sequence of positional arguments for the function.
        - A dictionary of keyword arguments for the function.

        Therefore, each function `f` in `function` should be callable
        with the corresponding generator `g` in `input_gen` as follows:
        ```python
        args, kwargs = g(input_size)
        f(*args, **kwargs)
        ```
    function_name
        Name(s) of the function(s) to be profiled.
        If not provided, the function's module and qualified name will be used.
        If the same function is provided multiple times,
        the name will be suffixed with an index to ensure uniqueness.
    output_evaluator
        Optional callable that evaluates the outputs of the functions.
        This can be used to validate or process the outputs in parallel to profiling.
        If not provided, no output evaluation will be performed.
        The evaluator function should accept
        the following parameters (passed as positional arguments in this order):
        - `input_size`: The input size that was used.
        - `input_args`: A sequence of positional arguments used for each function.
        - `input_kwargs`: A sequence of keyword arguments used for each function.
        - `outputs`: A sequence of outputs returned by each function.
    """

    def __init__(
        self,
        function: Callable | Sequence[Callable],
        input_gen: ArgGen | Sequence[ArgGen],
        function_name: Sequence[str] | None = None,
        output_evaluator: Callable[[InputSize, Sequence[InputArgs], Sequence[InputKwargs], Sequence[Output]], None] | None = None,
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

        self._input_sizes: list[int] = None
        self._runs: int = None
        self._loops_per_run: int = None
        self._results: np.ndarray = None
        return

    def profile(
        self,
        input_sizes: Sequence[InputSize],
        runs: int = 100,
        loops_per_run: int = 1,
    ) -> np.ndarray:
        """Profile all functions.

        Parameters
        ----------
        input_sizes
            Different input sizes to profile the functions with.
        runs
            Number of times the profiling is repeated for each input size.
            Higher values provide more accurate results.
            The shortest duration between all runs will be selected,
            as it represents the most accurate duration.
        loops_per_run
            Number of times each run is repeated.
            For each run, the average of all loops will be selected.

        Returns
        -------
        A 2D array with the shape `(n_functions, n_input_sizes)`,
        containing the lowest time taken for each function at each input size.
        """
        self._input_sizes = input_sizes
        self._runs = runs
        self._loops_per_run = loops_per_run
        self._results = np.empty((len(self._funcs), len(input_sizes)), dtype=float)
        for input_size_idx, input_size in enumerate(input_sizes):
            input_args = []
            input_kwargs = []
            outputs = []
            for func_idx, (func, arg_gen) in enumerate(zip(self._funcs, self._input_gens, strict=True)):
                args, kwargs = arg_gen(input_size)
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
                self._results[func_idx, input_size_idx] = shortest_run_time
            if self._output_eval is not None:
                self._output_eval(input_size, input_args, input_kwargs, outputs)
        return self._results

    def plot(self, show: bool = True) -> tuple[plt.Figure, plt.Axes]:
        """Plot the profiling results.

        This function must be called after `profile` has been executed.
        It plots the latest profiling results for each function
        against the input sizes used during profiling.

        Parameters
        ----------
        show
            Call `plt.show()` after plotting.

        Returns
        -------
        A tuple containing the figure and axes of the plot.
        """
        if self._results is None:
            raise ValueError(
                "No profiling has been performed yet; call `FunctionProfiler.profile` first."
            )
        fig, ax = plt.subplots()
        artists = []
        for func_name, result in zip(self._func_names, self._results, strict=True):
            (line,) = ax.plot(
                self._input_sizes,
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
