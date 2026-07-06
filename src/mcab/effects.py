from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Callable, Literal, Any, Dict, Type
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .transform import Linearizer

__all__ = [
    'PercentEffectSizer',
    'ConstantEffectSizer',
    'CustomEffectSizer',
    'FunctionEffectSizer',
    '_EFFECTS_DICT',
]

class RatioType(IntEnum):
    """Enumeration for ratio data types.
    
    Attributes:
        INT: Integer ratio type
        FLOAT: Float ratio type
    """
    INT   = 0
    FLOAT = 1
    LIN   = 2

RatioTypeValues: set[int] = set(map(int, RatioType))

_RatioTypes: dict[Union[str, bool], Union[RatioType, bool]] = {
    'int':   RatioType.INT,
    'float': RatioType.FLOAT,
    'lin':   RatioType.LIN,
    False: False
}

class EffectSizer(ABC):
    """Abstract base class for effect sizing strategies.
    
    This class provides the foundation for different effect sizing approaches
    in A/B testing simulations. It handles both continuous and ratio metrics,
    with support for transformations and covariates.
    
    Attributes:
        effect: The effect size to apply
        proportion: Whether to treat data as proportions (0-1)
        ratio: Type of ratio metric ('int', 'float', or False for non-ratio)
        ratio_call: Whether ratio mode is enabled
        rng: Random number generator
    """
    def __init__(
        self,
        proportion: bool = False,
        ratio: Literal['int', 'float', 'lin', False] = False,
        seed: int = 42,
        func: Optional[Callable] = None,
        custom_data: Optional[Union[NDArray, Tuple[NDArray, NDArray]]] = None,
        raw_data: Optional[Union[NDArray, Tuple[NDArray, NDArray]]] = None
    ) -> None:
        """Initialize the EffectSizer.
        
        Args:
            proportion: If True, treat data as proportions (0-1 bounded)
            ratio: Type of ratio metric - 'int' for integer counts, 'float' for continuous ratios, 'lin' for linearized ratios, or False for non-ratio metrics
            seed: Seed used to build the internal random number generator
            func: Optional user-supplied callable (used by FunctionEffectSizer)
            custom_data: Optional pre-computed data with effects (used by CustomEffectSizer)
            raw_data: Optional original data without effects (used by CustomEffectSizer)
            
        Raises:
            ValueError: If ratio is not one of 'int', 'float', 'lin', or False
        """
        if ratio not in _RatioTypes:
            raise ValueError('ratio must be one of "int", "float", or False.')
        self.proportion = proportion
        self.ratio = _RatioTypes[ratio]
        self.ratio_call = ratio != False
        self.rng = np.random.default_rng(seed)
        self.func = func
        self.custom_data = custom_data
        self.raw_data = raw_data

    def __call__(
        self,
        raw_data: Union[NDArray, Tuple[NDArray, NDArray]],
        effect: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Apply effect to the data.
        
        Args:
            raw_data: Raw data array or tuple of (numerator, denominator) for ratio metrics
            effect: Optional effect size to apply (no effect if None or 0)
            seed: Optional seed forwarded to the underlying effect application for reproducibility
            
        Returns:
            Data with effect applied, same structure as raw_data
        """
        if self.ratio_call:
            if self.ratio == RatioType.LIN:
                return self._call_ratio_lin(
                    effect=effect,
                    raw_data=raw_data,
                    ratio=self.ratio,
                    seed=seed
                )
            # Forward `seed` so the ratio INT stochastic-rounding path below is
            # reproducible. Without it `_call_ratio` falls into the `seed is
            # None` branch and consumes the shared stateful `self.rng`, which,
            # under joblib parallelism (n_jobs != 1), is advanced by worker
            # processes in a non-reproducible interleaving and makes the
            # measured ratio power drift between identical reruns.
            return self._call_ratio(
                effect=effect,
                raw_data=raw_data,
                ratio=self.ratio,
                seed=seed
            )
        else:
            return self._call_iid(
                effect=effect,
                raw_data=raw_data,
                seed=seed
            )

    def _call_iid(
        self,
        effect: float,
        raw_data: NDArray,
        seed: Optional[int] = None
    ) -> NDArray:
        """Apply effect to independent and identically distributed (IID) data.
        
        Args:
            effect: The effect size to apply
            raw_data: Raw data array
            seed: Optional seed forwarded to the proportion-flipping logic for reproducibility
            
        Returns:
            Data with effect applied
        """
        # Target is continuous
        if not effect:
            effect_abs = 0.0
        elif self.proportion:
            effect_abs = self._effect_proportion_iid(
                effect=effect,
                raw_data=raw_data,
                seed=seed
            )
        else:
            effect_abs = self._effect_continous_iid(
                effect=effect,
                raw_data=raw_data
            )

        data_with_effect = raw_data + effect_abs

        return data_with_effect

    def _call_ratio(
        self,
        effect: float,
        raw_data: Tuple[NDArray, NDArray],
        ratio: Union[RatioType, int] = RatioType.INT,
        seed: Optional[int] = None
    ) -> Tuple[NDArray, NDArray]:
        """Apply effect to ratio metrics (numerator/denominator).
        
        Args:
            effect: The effect size to apply
            raw_data: Tuple of (numerator, denominator) arrays
            ratio: Ratio type (INT or FLOAT)
            seed: Optional seed forwarded to the integer stochastic-rounding step for reproducibility
            
        Returns:
            Tuple of (numerator_with_effect, denominator)
        """
        
        num_raw, denom = raw_data
        
        if effect:
            num_effect = self._effect_continous_iid(
                effect=effect,
                raw_data=num_raw
            )
        else:
            num_effect = 0.0

        num_with_effect = np.maximum(num_raw + num_effect, 0)
        if self.proportion:
            num_with_effect = np.minimum(num_with_effect, denom)

        if ratio == RatioType.INT:
            # Stochastic (unbiased) rounding instead of truncation.
            #
            # Plain `.astype(int)` truncates toward zero. For count-style ratio
            # metrics (e.g. CTR) the per-user numerator is tiny (0, 1, 2, ...),
            # so a small multiplicative effect like 1 * (1 + 0.05) = 1.05 is
            # floored straight back to 1. The effect is wiped out for almost
            # every user, the AB split becomes indistinguishable from an AA
            # split, and the measured power collapses to ~alpha for any effect.
            #
            # Stochastic rounding maps a value x to floor(x) with probability
            # 1 - frac(x) and to floor(x) + 1 with probability frac(x), so the
            # numerator stays integer while E[rounded] == x. This preserves the
            # injected effect on average and yields a correct power curve.
            if seed is None:
                rng = self.rng
            else:
                rng = np.random.default_rng(seed)
            floor = np.floor(num_with_effect)
            frac = num_with_effect - floor
            num_with_effect = (floor + (rng.random(num_with_effect.shape) < frac)).astype(int)

        return num_with_effect, denom
    
    def _call_ratio_lin(
        self,
        effect: float,
        raw_data: Tuple[NDArray, NDArray],
        ratio: Union[RatioType, int] = RatioType.INT,
        seed: Optional[int] = None
    ):
        test_raw, control_raw = raw_data
        test, control, theta = Linearizer(test_raw, control_raw)

        if effect == 0:
            data = np.concatenate([test, control])
            return data
        
        # Forward `seed` so the integer stochastic rounding inside _call_ratio
        # is reproducible under parallel execution (see __call__ note).
        test_mod_num, test_denom = self._call_ratio(
            effect=effect,
            raw_data=test_raw,
            ratio=ratio,
            seed=seed
        )

        test_linearized = test_mod_num - theta * test_denom
        return np.concatenate((test_linearized, control))
    
    def _effect_proportion_iid(
        self,
        effect: float,
        raw_data: NDArray,
        seed: Optional[int] = None
    ) -> NDArray:
        """Calculate effect for binary proportion data.
        
        This method flips binary values (0/1) to achieve the target proportion
        after applying the effect.
        
        Args:
            effect: The effect size to apply
            raw_data: Binary data array (0s and 1s)
            seed: Optional seed forwarded to the flip-selection RNG for reproducibility
            
        Returns:
            Array of changes to apply to raw_data
        """
        if not effect:
            return np.zeros_like(raw_data)
        nobs = raw_data.shape[0]
        
        current_rate: float = raw_data.mean()
        target_rate: float = max(min(current_rate + self._effect_continous_iid(effect, current_rate), 1.0), 0.0)

        if target_rate == 0.0:
            return -raw_data.copy()
        if target_rate == 1.0:
            return np.ones_like(raw_data) - raw_data
        
        req_ones: int = round(nobs * target_rate)
        fact_ones: int = (raw_data == 1).sum()
        modified: NDArray = raw_data.copy()

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if req_ones > fact_ones:
            # Need more ones: flip some zeros to ones
            n_flip: int = req_ones - fact_ones
            zero_positions: NDArray = np.where(modified == 0)[0]
            if n_flip > len(zero_positions):
                # All zeros already flipped — clip to maximum possible
                modified[zero_positions] = 1
            else:
                flip_positions: NDArray = rng.choice(zero_positions, size=n_flip, replace=False)
                modified[flip_positions] = 1

        elif req_ones < fact_ones:
            # Need fewer ones: flip some ones to zeros
            n_flip: int = fact_ones - req_ones
            one_positions: NDArray = np.where(modified == 1)[0]
            if n_flip > len(one_positions):
                modified[one_positions] = 0
            else:
                flip_positions: NDArray = rng.choice(one_positions, size=n_flip, replace=False)
                modified[flip_positions] = 0

        return modified - raw_data

    @abstractmethod
    def _effect_continous_iid(
        self,
        effect: float,
        raw_data: Union[NDArray, float]
    ) -> Union[NDArray, float]:
        """Calculate continuous effect for IID data.
        
        This abstract method must be implemented by subclasses to define
        how the effect is calculated for continuous metrics.
        
        Args:
            effect: The effect size to apply
            raw_data: Raw data array or scalar value
            
        Returns:
            Calculated effect (same type as raw_data)
        """
        pass


class PercentEffectSizer(EffectSizer):
    """Effect sizer that applies percentage-based effects.
    
    The effect is calculated as a percentage of the raw data value.
    For example, effect=0.1 means a 10% increase.
    """

    def _effect_continous_iid(
        self,
        effect: float,
        raw_data: Union[NDArray, float]
    ) -> Union[NDArray, float]:
        """Calculate effect as percentage of raw data.
        
        Args:
            effect: The percentage effect (e.g., 0.1 for 10% increase)
            raw_data: Raw data array or scalar
            
        Returns:
            Effect calculated as raw_data * effect
        """
        return raw_data * effect
    
class ConstantEffectSizer(EffectSizer):
    """Effect sizer that applies constant (absolute) effects.
    
    The effect is a fixed value added to the data, independent of the
    original data values.
    """
    
    def _effect_continous_iid(
        self,
        effect: float,
        raw_data: Union[NDArray, float]
    ) -> float:
        """Calculate constant effect.
        
        Args:
            effect: The constant effect value to add
            raw_data: Raw data array or scalar (not used in calculation)
            
        Returns:
            The constant effect value
        """
        return effect
    
class CustomEffectSizer(EffectSizer):
    """Custom effect sizer using pre-computed data with effects.
    
    This class allows using pre-generated data with effects already applied,
    useful for complex effect patterns that don't fit standard formulas.
    
    Attributes:
        raw_data: Original data without effects
        data_with_effect: Data with effects pre-applied
        ratio: Whether data is in ratio format
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # custom_data and raw_data must both be provided.
        if self.custom_data is None:
            raise ValueError(
                "CustomEffectSizer: custom_data must not be None."
            )
        if self.raw_data is None:
            raise ValueError(
                "CustomEffectSizer: raw_data must not be None."
            )

        # Shapes of custom_data and raw_data must match.
        def _shape(x):
            if isinstance(x, tuple):
                return x[0].shape
            return np.asarray(x).shape

        cd_shape = _shape(self.custom_data)
        rd_shape = _shape(self.raw_data)
        if cd_shape != rd_shape:
            raise ValueError(
                f"CustomEffectSizer: custom_data and raw_data must have the same shape, "
                f"but got custom_data.shape={cd_shape} and raw_data.shape={rd_shape}."
            )

        # For iid proportion data (binary 0/1), custom_data must contain
        # only 0s and 1s — otherwise the simulations will silently produce
        # incorrect p-values (e.g., a proportion test on non-binary data).
        if self.proportion and not self.ratio_call:
            data = np.asarray(self.custom_data)
            if not np.all((data == 0) | (data == 1)):
                raise ValueError(
                    "CustomEffectSizer: proportion=True and ratio=False requires "
                    "custom_data to contain only 0s and 1s, "
                    f"but got unique values: {np.unique(data)}"
                )

        # For ratio proportion data (CTR-like), numerator must not exceed
        # denominator in either raw_data or custom_data.
        if self.proportion and self.ratio_call:
            cd_num, cd_denom = self.custom_data
            if not np.all(cd_num <= cd_denom):
                raise ValueError(
                    "CustomEffectSizer: proportion=True and ratio metric requires "
                    "custom_data numerator <= denominator for all observations."
                )
            rd_num, rd_denom = self.raw_data
            if not np.all(rd_num <= rd_denom):
                raise ValueError(
                    "CustomEffectSizer: proportion=True and ratio metric requires "
                    "raw_data numerator <= denominator for all observations."
                )

    def __call__(
        self,
        idx: Union[int, slice, NDArray],
        effect=True,
        seed=None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Get data with or without effect applied.
        
        Args:
            idx: Index or slice to select data
            effect: If truthy, return data_with_effect; otherwise return raw_data
            seed: Not used (kept for API consistency with EffectSizer.__call__)
            
        Returns:
            Selected data with or without effect
        """
        data = self.custom_data if effect else self.raw_data
        if isinstance(data, tuple):
            return (data[0][idx], data[1][idx])
        return data[idx]
    
    def _effect_continous_iid(
        self,
        effect: float,
        raw_data: Union[NDArray, float]
    ) -> None:
        """Not implemented for CustomEffectSizer.
        
        Args:
            effect: Effect size (not used)
            raw_data: Raw data (not used)
        """
        pass

class FunctionEffectSizer(EffectSizer):
    """Effect sizer that applies a user-supplied function.

    The effect is produced by a custom callable that receives the test and
    control samples together with the shuffled index and returns the test
    sample with the effect applied. When ``effect`` is falsy the original
    (untouched) test sample is returned instead.

    To protect the experiment data, every array handed to ``func`` is passed as
    a READ-ONLY view: any in-place mutation inside ``func`` raises a
    ``ValueError`` instead of silently corrupting the shared
    ``target.data_raw``. The user function is therefore forced to return a NEW
    array with the effect applied.

    Attributes:
        func: User-supplied callable
            ``func(test, control, test_idx, control_idx, idx, seed) -> test_mod``.
    """

    @staticmethod
    def _as_readonly(
        data: Union[NDArray, Tuple[NDArray, ...]]
    ) -> Union[NDArray, Tuple[NDArray, ...]]:
        """Return a read-only view of an array (or tuple of arrays).

        A view (not a copy) is returned, so it stays cheap, but its
        ``writeable`` flag is cleared. Any attempt to mutate it in place raises
        ``ValueError: assignment destination is read-only``, which prevents the
        user function from modifying the shared experiment data.

        Args:
            data: Array, or tuple of arrays (e.g. ``(num, denom)`` for ratios).

        Returns:
            A read-only view with the same structure as ``data``.
        """
        if isinstance(data, tuple):
            return tuple(FunctionEffectSizer._as_readonly(x) for x in data)
        if isinstance(data, np.ndarray):
            view = data.view()
            view.flags.writeable = False
            return view
        return data

    def __call__(
        self,
        first_argument: Tuple[Any, Any, Any, NDArray, NDArray, NDArray],
        effect: Any = True,
        seed: Optional[int] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Apply the user function to the test sample.

        Args:
            first_argument: Tuple
                ``(test, control, test_control, test_idx, control_idx, test_control_idx)``.
                ``test`` and ``control`` are the raw test/control samples
                (arrays for iid metrics or ``(num, denom)`` tuples for ratio
                metrics). ``test_control`` is the full dataset (test + control
                combined, same structure as test/control). ``test_idx`` /
                ``control_idx`` are the per-group index arrays and
                ``test_control_idx`` is the full shuffled index array. All of
                them are passed to ``func`` as read-only views.
            effect: If truthy, returns
                ``func(test, control, test_control, test_idx, control_idx, test_control_idx, seed)``;
                otherwise returns the original (untouched) test sample.
            seed: Forwarded to ``func`` for reproducibility.

        Returns:
            Test sample with the effect applied, or the original test sample.
        """
        test, control, test_control, test_idx, control_idx, test_control_idx = first_argument
        if not effect:
            return test
        return self.func(
            test=self._as_readonly(test),
            control=self._as_readonly(control),
            test_control=self._as_readonly(test_control),
            test_idx=self._as_readonly(test_idx),
            control_idx=self._as_readonly(control_idx),
            test_control_idx=self._as_readonly(test_control_idx),
            seed=seed
        )

    def _effect_continous_iid(
        self,
        effect: float,
        raw_data: Union[NDArray, float]
    ) -> None:
        """Not implemented for FunctionEffectSizer.

        Args:
            effect: Effect size (not used)
            raw_data: Raw data (not used)
        """
        pass

_EFFECTS_DICT: Dict[str, Type[EffectSizer]] = {
    'percent':PercentEffectSizer,
    'const':ConstantEffectSizer,
    'custom':CustomEffectSizer,
    'function':FunctionEffectSizer
}
