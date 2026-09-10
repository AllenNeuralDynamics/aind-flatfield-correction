"""
Shared test doubles for the flatfield estimation tests.

Every seam these stand in for is patched explicitly by the tests, so the
suite behaves the same whether or not the real ``basicpy`` and
``aind-large-scale-prediction`` are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LazyPlane:
    """A computed slice of :class:`LazyArray`."""

    def __init__(self, values: np.ndarray) -> None:
        """
        Hold the already-materialized values.

        Parameters
        ----------
        values : np.ndarray
            Values the fake read returns.

        Returns
        -------
        None
        """
        self._values = values

    def compute(self) -> np.ndarray:
        """
        Return the values, standing in for a dask compute.

        Returns
        -------
        np.ndarray
            The held values.
        """
        return self._values


class LazyArray:
    """Minimal stand-in for the dask array ``open_tile`` returns."""

    def __init__(self, values: np.ndarray) -> None:
        """
        Wrap an in-memory array.

        Parameters
        ----------
        values : np.ndarray
            Array to serve, of any dimensionality.

        Returns
        -------
        None
        """
        self._values = values

    @property
    def ndim(self) -> int:
        """
        Number of dimensions.

        Returns
        -------
        int
            Dimension count of the wrapped array.
        """
        return self._values.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape.

        Returns
        -------
        tuple of int
            Shape of the wrapped array.
        """
        return self._values.shape

    def __getitem__(self, key: Any) -> Any:
        """
        Index the array the way the loader does.

        A scalar index drops an axis and returns another lazy array, as
        the squeeze loop in ``open_tile`` expects; anything else returns
        a computable plane selection.

        Parameters
        ----------
        key : Any
            Index or index array.

        Returns
        -------
        LazyArray or LazyPlane
            A further lazy view, or a computable selection.
        """
        if isinstance(key, int):
            return LazyArray(self._values[key])
        return LazyPlane(self._values[key])


def make_basic(
    fail: bool = False,
    record: list[dict[str, Any]] | None = None,
    span: float = 0.5,
):
    """
    Build a fake ``BaSiC`` class.

    The fitted flatfield is a ramp of width ``span`` around 1.0, which
    clears the plausibility guard, and ``transform`` spreads the images
    by ``smoothness_flatfield`` so that different candidates score
    different entropies -- smaller smoothness gives a narrower spread
    and therefore the lower (winning) entropy.

    Parameters
    ----------
    fail : bool, optional
        Raise from ``fit`` instead of fitting, by default False.
    record : list of dict, optional
        List that every constructor's keyword arguments are appended to.
    span : float, optional
        Peak-to-peak width of the fitted flatfield, by default 0.5.

    Returns
    -------
    type
        A class usable in place of ``basicpy.BaSiC``.
    """

    class FakeBaSiC:
        """Stand-in for ``basicpy.BaSiC``."""

        def __init__(self, **kwargs: Any) -> None:
            """
            Record the configuration it was built with.

            Parameters
            ----------
            **kwargs : Any
                BaSiC keyword arguments.

            Returns
            -------
            None
            """
            self.kwargs = kwargs
            self.flatfield = np.ones((1, 1), dtype=np.float32)
            self.darkfield = np.zeros((1, 1), dtype=np.float32)
            if record is not None:
                record.append(dict(kwargs))

        def fit(self, images: Any = None) -> None:
            """
            Derive a flatfield from the images' plane shape.

            Parameters
            ----------
            images : Any, optional
                Image stack to fit.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If the fake was built with ``fail=True``.
            """
            if fail:
                raise ValueError("fake fit failure")
            height, width = np.asarray(images).shape[-2:]
            ramp = np.linspace(1.0 - span / 2, 1.0 + span / 2, height * width)
            self.flatfield = ramp.reshape(height, width).astype(np.float32)
            self.darkfield = np.zeros((height, width), dtype=np.float32)

        def transform(self, images: Any) -> np.ndarray:
            """
            Spread the images by this candidate's smoothness.

            Parameters
            ----------
            images : Any
                Image stack to transform.

            Returns
            -------
            np.ndarray
                The images plus deterministic noise scaled by
                ``smoothness_flatfield``.
            """
            values = np.asarray(images, dtype=np.float64)
            scale = float(self.kwargs.get("smoothness_flatfield", 1.0))
            rng = np.random.default_rng(int(scale * 1000) % 9973)
            return values + rng.normal(scale=scale, size=values.shape)

    return FakeBaSiC


class InlineFuture:
    """A future whose result is already known."""

    def __init__(self, result: Any) -> None:
        """
        Hold a completed result.

        Parameters
        ----------
        result : Any
            Value to hand back.

        Returns
        -------
        None
        """
        self._result = result

    def result(self) -> Any:
        """
        Return the completed result.

        Returns
        -------
        Any
            The held value.
        """
        return self._result


class InlineExecutor:
    """Stand-in for ``ProcessPoolExecutor`` that runs work inline.

    Keeps the pool code paths under test without spawning processes,
    which would need the workers to re-import the patched seams.
    """

    def __init__(self, max_workers: int = 1, mp_context: Any = None) -> None:
        """
        Accept and ignore the pool's configuration.

        Parameters
        ----------
        max_workers : int, optional
            Ignored, by default 1.
        mp_context : Any, optional
            Ignored, by default None.

        Returns
        -------
        None
        """
        self.max_workers = max_workers
        self.mp_context = mp_context

    def __enter__(self) -> "InlineExecutor":
        """
        Enter the context manager.

        Returns
        -------
        InlineExecutor
            This executor.
        """
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        """
        Leave the context manager without swallowing exceptions.

        Parameters
        ----------
        *exc_info : Any
            Exception information, unused.

        Returns
        -------
        bool
            False, so exceptions propagate.
        """
        return False

    def submit(self, func: Any, *args: Any, **kwargs: Any) -> InlineFuture:
        """
        Run ``func`` immediately.

        Parameters
        ----------
        func : Any
            Callable to run.
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        InlineFuture
            A future already holding the return value.
        """
        return InlineFuture(func(*args, **kwargs))


def inline_as_completed(futures: Any) -> list[Any]:
    """
    Stand in for ``as_completed`` over :class:`InlineFuture` objects.

    Parameters
    ----------
    futures : Any
        Iterable (or dict) of inline futures.

    Returns
    -------
    list
        The futures, in submission order.
    """
    return list(futures)
