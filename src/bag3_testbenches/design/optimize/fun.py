# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This module defines special interpolation function classes that are used for the optimization based design flow."""

from copy import deepcopy
from typing import Optional, Dict, Union, Tuple, List, Callable, Any

import numpy as np

from bag.math.dfun import DiffFunction
from bag.math.interpolate import LinearInterpolator


class ConstraintMixin:
    """A mixin class that creates constraints in a dictionary format compatible with
    scipy.optimize.minimize's constraints parameter.

    It is meant to be used in multiple inheritance with DiffFunction.
    """

    # TODO: implement jacobian?

    def __eq__(self, other: Union[float, int, DiffFunction]) -> Dict[str, Any]:
        if isinstance(other, (float, int, DiffFunction)):
            return dict(type='eq', fun=self - other)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __ge__(self, other: Union[float, int, DiffFunction]) -> Dict[str, Any]:
        if isinstance(other, (float, int, DiffFunction)):
            return dict(type='ineq', fun=self - other)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __le__(self, other: Union[float, int, DiffFunction]) -> Dict[str, Any]:
        if isinstance(other, (float, int, DiffFunction)):
            return dict(type='ineq', fun=other - self)
        else:
            raise NotImplementedError('Unknown type %s' % type(other))

    def __gt__(self, other: Union[float, int, DiffFunction]) -> Dict[str, Any]:
        return self.__ge__(other)

    def __lt__(self, other: Union[float, int, DiffFunction]) -> Dict[str, Any]:
        return self.__le__(other)


class ArgMapFunction(ConstraintMixin, DiffFunction):
    """A DiffFunction that supports optimization constraints and argument mapping.

    Parameters
    ----------
    fn : DiffFunction
        the inner/parent function.
    idx_map : Optional[Dict[int, int]]
        the dictionary mapping the inner function's argument indices to outer argument indices.
        Defaults to None.
    fixed_vals : Optional[Dict[int, Union[int, float]]]
        the dictionary mapping the inner function's argument indices to their fixed values.
        Defaults to None.
    input_bounds : Optional[Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]]
        the dictionary mapping the inner function's argument indices to their new bounds/ranges.
        These bounds are intersected with the inner function's input ranges to get the new input ranges.
        Defaults to None.
    """
    def __init__(self, fn: DiffFunction, idx_map: Optional[Dict[int, int]] = None,
                 fixed_vals: Optional[Dict[int, Union[int, float]]] = None,
                 input_bounds: Optional[
                     Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]] = None):
        self._fn = fn
        # TODO: should it be possible for multiple indices to map to the same index?
        self._idx_map = deepcopy(idx_map) or {}
        self._fixed_vals = fixed_vals or {}
        self._input_bounds = input_bounds or {}

        # Initialize idx_map by defaulting to identity
        for idx in range(self._fn.ndim):
            if idx in self._fixed_vals:
                if idx in self._idx_map:
                    raise ValueError(f"Index {idx} cannot be defined for both idx_map and fixed_vals")
            elif idx not in self._idx_map:
                self._idx_map[idx] = idx

        # Outer dimension is set by the highest index used in idx_map
        ndim = max(self._idx_map.values()) + 1

        # Compute input_ranges and delta_list by using the index mapping
        input_ranges = [(None, None) for _ in range(ndim)]
        delta_list = None if fn.delta_list is None else [None for _ in range(ndim)]
        for idx in range(self.inner_ndim):
            if idx in self._fixed_vals:  # no range or delta since argument is constrained to a fixed value
                continue
            idx_mapped = self._idx_map[idx]
            in_range = fn.input_ranges[idx]
            if idx in self._input_bounds:  # compute the intersection of the two ranges
                bnd_l, bnd_h = self._input_bounds[idx]
                bnd_l = -np.inf if bnd_l is None else bnd_l
                bnd_h = np.inf if bnd_h is None else bnd_h
                in_range = (max(bnd_l, in_range[0]), min(bnd_h, in_range[1]))
            input_ranges[idx_mapped] = in_range
            if delta_list is not None:
                delta_list[idx_mapped] = fn.delta_list[idx]

        super().__init__(input_ranges, delta_list)

    @property
    def inner_ndim(self) -> int:
        """ Return the number of dimensions for the inner function."""
        return self._fn.ndim

    def _get_args(self, xi: np.ndarray) -> np.ndarray:
        """Applies argument mapping.

        Parameters
        ----------
        xi : np.ndarray
            The input coordinates, with shape (..., ndim)

        Returns
        -------
        new_xi : np.ndarray
            The output coordinates, with shape (..., inner_ndim).
        """
        # noinspection PyTypeChecker
        new_xi = np.full(tuple(xi.shape[:-1]) + (self.inner_ndim, ), np.nan)
        for i in range(self.inner_ndim):
            if i in self._fixed_vals:
                new_xi[..., i] = self._fixed_vals[i]
            else:
                new_xi[..., i] = xi[..., self._idx_map[i]]
        return new_xi

    def __call__(self, xi: np.ndarray) -> np.ndarray:
        return self._fn(self._get_args(np.array(xi)))

    # TODO: verify below methods
    def deriv(self, xi: np.ndarray, j: int) -> np.ndarray:
        return self._fn.deriv(self._get_args(np.array(xi)), j)

    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        return self._fn.jacobian(self._get_args(np.array(xi)))


class ArgMapNormFunction(ArgMapFunction):
    """An ArgMapFunction that supports input (de-)normalization.
    Optimizers are sensitive to the input range.

    Parameters
    ----------
    fn : DiffFunction
        the inner function.
    idx_map : Optional[Dict[int, int]]
        the dictionary mapping the inner function's argument indices to outer argument indices.
        Defaults to None.
    fixed_vals : Optional[Dict[int, Union[int, float]]]
        the dictionary mapping the inner function's argument indices to their fixed values.
        Defaults to None.
    input_bounds : Optional[Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]]
        the dictionary mapping the inner function's argument indices to their new bounds/ranges.
        These bounds are intersected with the inner function's input ranges to get the new input ranges.
        Defaults to None.
    """
    def __init__(self, fn: DiffFunction,
                 idx_map: Optional[Dict[int, int]] = None, fixed_vals: Optional[Dict[int, Union[int, float]]] = None,
                 input_bounds: Optional[
                     Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]] = None):
        super().__init__(fn, idx_map, fixed_vals, input_bounds)

        # Get parent function's scale_list
        if isinstance(fn, LinearInterpolator):
            fn: LinearInterpolator
            self._fn_scale_list = [(pvec[0], pvec[1] - pvec[0]) for pvec in fn._points]
        elif hasattr(fn, '_scale_list'):
            self._fn_scale_list = fn._scale_list
        elif hasattr(fn, '_points'):
            raise NotImplementedError("See developer")
        else:  # Approximate scale_list by taking the input range
            self._fn_scale_list = [(bnd_l, bnd_h - bnd_l) for (bnd_l, bnd_h) in fn.input_ranges]

        self._fn_offset_list, self._fn_spacing_list = map(np.array, zip(*self._fn_scale_list))

        self._fixed_vals_norm = {idx: self.norm_inner_input_elem(val, idx)
                                 for idx, val in self._fixed_vals.items()}

        # Compute scale_list post-normalization
        scale_list = [(0, 1) for _ in range(self.ndim)]
        for idx in range(self.inner_ndim):
            if idx in self._fixed_vals:
                continue
            scale_list[self._idx_map[idx]] = self._fn_scale_list[idx]

        self._scale_list = scale_list
        self._offset_list, self._spacing_list = map(np.array, zip(*self._scale_list))

        self._input_ranges_norm = []
        for idx, (bnd_l, bnd_h) in enumerate(self._input_ranges):
            bnd_l_norm = None if bnd_l is None else self.norm_input_elem(bnd_l, idx)
            bnd_h_norm = None if bnd_h is None else self.norm_input_elem(bnd_h, idx)
            self._input_ranges_norm.append((bnd_l_norm, bnd_h_norm))

    @property
    def input_ranges_norm(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """ Return the normalized input ranges."""
        return self._input_ranges_norm

    def denorm_inner_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """De-normalizes the inner function's input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the inner function's input element to de-normalize
        idx : int
            the index to de-normalize on

        Returns
        -------
        ans : np.ndarray
            The de-normalized inner function's input element.
        """
        return _denorm_input_elem_helper(val, idx, self._fn_offset_list, self._fn_spacing_list)

    def norm_inner_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """Normalizes the inner function's input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the inner function's input element to normalize
        idx : int
            the index to normalize on

        Returns
        -------
        ans : np.ndarray
            The normalized inner function's input element.
        """
        return _norm_input_elem_helper(val, idx, self._fn_offset_list, self._fn_spacing_list)

    def denorm_inner_input(self, val: np.ndarray) -> np.ndarray:
        """De-normalizes the inner function's input vector.

        Parameters
        ----------
        val : np.ndarray
            the inner function's input vector to de-normalize

        Returns
        -------
        ans : np.ndarray
            The de-normalized inner function's input vector.
        """
        return _denorm_input_helper(val, self._fn_offset_list, self._fn_spacing_list)

    def norm_inner_input(self, val: np.ndarray) -> np.ndarray:
        """Normalizes the inner function's input vector.

        Parameters
        ----------
        val : np.ndarray
            the inner function's input vector to normalize

        Returns
        -------
        ans : np.ndarray
            The normalized inner function's input vector.
        """
        return _norm_input_helper(val, self._fn_offset_list, self._fn_spacing_list)

    def denorm_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """De-normalizes the input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the input element to de-normalize
        idx : int
            the index to de-normalize on

        Returns
        -------
        ans : np.ndarray
            The de-normalized input element.
        """
        return _denorm_input_elem_helper(val, idx, self._offset_list, self._spacing_list)

    def norm_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """Normalizes the input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the input element to normalize
        idx : int
            the index to normalize on

        Returns
        -------
        ans : np.ndarray
            The normalized input element.
        """
        return _norm_input_elem_helper(val, idx, self._offset_list, self._spacing_list)

    def denorm_input(self, val: np.ndarray) -> np.ndarray:
        """De-normalizes the input vector.

        Parameters
        ----------
        val : np.ndarray
            the input vector to de-normalize

        Returns
        -------
        ans : np.ndarray
            The de-normalized input vector.
        """
        return _denorm_input_helper(val, self._offset_list, self._spacing_list)

    def norm_input(self, val: np.ndarray) -> np.ndarray:
        """Normalizes the input vector.

        Parameters
        ----------
        val : np.ndarray
            the input vector to normalize

        Returns
        -------
        ans : np.ndarray
            The normalized input vector.
        """
        return _norm_input_helper(val, self._offset_list, self._spacing_list)

    def _get_args(self, xi: np.ndarray, is_norm_input: bool = True):
        """Applies argument mapping and applies de-normalization (if applicable).

        Parameters
        ----------
        xi : np.ndarray
            The input coordinates, with shape (..., ndim)
        is_norm_input : bool
            True if the input is normalized. Default is True.

        Returns
        -------
        new_xi : np.ndarray
            The output coordinates, with shape (..., inner_ndim).
        """
        # noinspection PyTypeChecker
        new_xi = np.full(tuple(xi.shape[:-1]) + (self.inner_ndim, ), np.nan)
        for i in range(self.inner_ndim):
            if i in self._fixed_vals:
                new_xi[..., i] = self._fixed_vals_norm[i] if is_norm_input else self._fixed_vals[i]
            else:
                new_xi[..., i] = xi[..., self._idx_map[i]]
        if is_norm_input:
            new_xi = self.denorm_inner_input(new_xi)
        return new_xi

    def __call__(self, xi: np.ndarray, is_norm_input: bool = True) -> np.ndarray:
        xi_mod = self._get_args(np.array(xi), is_norm_input)
        return self._fn(xi_mod)

    def deriv(self, xi: np.ndarray, j: int) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CustomVectorArgMapFunction(object):
    """An interpolating function that takes in multiple sub-functions and computes a vector.

    Parameters
    ----------
    fn_list : List[DiffFunction]
        the list of sub-functions. All sub-functions must have the same dimensions and input ranges.
    indep_idx_list : Optional[List[int]]
        the list of sub-function's argument indices for which the input value can be independently set per sub-function.
        Defaults to None.
    fixed_vals : Optional[Dict[int, Union[int, float, List[Union[int, float]], np.ndarray]]]
        the dictionary mapping the sub-function's argument indices to their fixed values.
        Defaults to None.
    input_bounds : Optional[Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]]
        the dictionary mapping the sub-function's argument indices to their new bounds/ranges.
        These bounds are intersected with the sub-function's input ranges to get the new input ranges.
        Defaults to None.
    """
    def __init__(self, fn_list: List[DiffFunction], indep_idx_list: Optional[List[int]] = None,
                 fixed_vals: Optional[Dict[int, Union[int, float, List[Union[int, float]], np.ndarray]]] = None,
                 input_bounds: Optional[
                     Dict[int, Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]]] = None):
        indep_idx_list = sorted(indep_idx_list or [])
        fixed_vals = fixed_vals or {}
        if not all(fn.input_ranges == fn_list[0].input_ranges for fn in fn_list[1:]):
            raise ValueError("Input ranges are not the same across all sub-functions")
        if not all(fn.ndim == fn_list[0].ndim for fn in fn_list[1:]):
            raise ValueError("Number of dimensions is not the same across all sub-functions")
        num_fn = len(fn_list)

        # Compute idx_map and fixed_vals for each sub function
        idx_map_list = [{} for _ in range(num_fn)]
        fixed_vals_list = [{} for _ in range(num_fn)]

        new_idx = 0
        for dim_idx in range(fn_list[0].ndim):
            if dim_idx in fixed_vals:
                val_arr = np.broadcast_to(fixed_vals[dim_idx], num_fn)
                for fn_idx, val in enumerate(val_arr):
                    fixed_vals_list[fn_idx][dim_idx] = val
            elif dim_idx in indep_idx_list:
                for fn_idx in range(num_fn):
                    idx_map_list[fn_idx][dim_idx] = new_idx + fn_idx
                new_idx += num_fn
            else:
                for fn_idx in range(num_fn):
                    idx_map_list[fn_idx][dim_idx] = new_idx
                new_idx += 1

        self._idx_map_list = idx_map_list
        self._fixed_vals_list = fixed_vals_list

        # Convert each sub-function to ArgMapNormFunction
        # TODO: independent input_bounds setting?
        self._fn_list = [ArgMapNormFunction(fn, _idx_map, _fixed_vals, input_bounds) for fn, _idx_map, _fixed_vals in
                         zip(fn_list, idx_map_list, fixed_vals_list)]

        self._fixed_vals = fixed_vals
        self._indep_idx_list = indep_idx_list

        # Construct input ranges
        self._input_ranges = []
        self._input_ranges_norm = []
        self._scale_list = []

        for dim_idx in range(self._fn_list[0].inner_ndim):
            if dim_idx in fixed_vals:
                continue
            if dim_idx in indep_idx_list:
                _fn_list = self._fn_list
            else:
                _fn_list = [self._fn_list[0]]
            for fn in _fn_list:
                self._input_ranges.append(fn.input_ranges[fn._idx_map[dim_idx]])
                self._input_ranges_norm.append(fn.input_ranges_norm[fn._idx_map[dim_idx]])
                self._scale_list.append(fn._scale_list[fn._idx_map[dim_idx]])
        self._offset_list, self._spacing_list = map(np.array, zip(*self._scale_list))

    @property
    def ndim(self) -> int:
        """Returns the number of input dimensions."""
        return len(self._scale_list)

    @property
    def num_fn(self) -> int:
        """Returns the number of sub-functions."""
        return len(self._fn_list)

    @property
    def input_ranges(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Returns the list of input ranges."""
        return self._input_ranges

    @property
    def input_ranges_norm(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Returns the list of normalized input ranges."""
        return self._input_ranges_norm

    @property
    def scale_list(self) -> List[Tuple[float, float]]:
        """Returns a list of (offset, spacing)."""
        return self._scale_list

    def __call__(self, xi: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([fn(xi, **kwargs) for fn in self._fn_list])

    def deriv(self, xi: np.ndarray, j: int) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def unmap_input(self, xi: np.ndarray) -> List[Union[float, np.ndarray]]:
        """Unmaps the input.

        Parameters
        ----------
        xi : np.ndarray
            The input coordinates.

        Returns
        -------
        rv : List[Union[float, np.ndarray]]
            The unmapped coordinates.
        """
        rv = []
        xi_idx = 0
        for dim_idx in range(self._fn_list[0].inner_ndim):
            if dim_idx in self._fixed_vals:
                rv.append(self._fixed_vals[dim_idx])
            elif dim_idx in self._indep_idx_list:
                rv.append(xi[xi_idx:xi_idx + self.num_fn])
                xi_idx += self.num_fn
            else:
                rv.append(xi[xi_idx])
                xi_idx += 1
        return rv

    def map_input(self, xi: List[Union[float, np.ndarray]]) -> np.ndarray:
        """Maps the input.

        Parameters
        ----------
        xi : List[Union[float, np.ndarray]]
            The input coordinates.

        Returns
        -------
        rv : np.ndarray
            The mapped coordinates.
        """
        # noinspection PyTypeChecker
        rv = np.full(self.ndim, np.nan)
        idx = 0
        for dim_idx, val in enumerate(xi):
            if dim_idx in self._fixed_vals:
                continue
            if dim_idx in self._indep_idx_list:
                for i in range(self.num_fn):
                    rv[idx] = xi[dim_idx][i]
                    idx += 1
            else:
                rv[idx] = xi[dim_idx]
                idx += 1
        return rv

    def denorm_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """De-normalizes the input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the input element to de-normalize
        idx : int
            the index to de-normalize on

        Returns
        -------
        ans : np.ndarray
            The de-normalized input element.
        """
        return _denorm_input_elem_helper(val, idx, self._offset_list, self._spacing_list)

    def norm_input_elem(self, val: Union[float, np.ndarray], idx: int) -> np.ndarray:
        """Normalizes the input element.

        Parameters
        ----------
        val : Union[float, np.ndarray]
            the input element to normalize
        idx : int
            the index to normalize on

        Returns
        -------
        ans : np.ndarray
            The normalized input element.
        """
        return _norm_input_elem_helper(val, idx, self._offset_list, self._spacing_list)

    def denorm_input(self, val: np.ndarray) -> np.ndarray:
        """De-normalizes the input vector.

        Parameters
        ----------
        val : np.ndarray
            the input vector to de-normalize

        Returns
        -------
        ans : np.ndarray
            The de-normalized input vector.
        """
        return _denorm_input_helper(val, self._offset_list, self._spacing_list)

    def norm_input(self, val: np.ndarray) -> np.ndarray:
        """Normalizes the input vector.

        Parameters
        ----------
        val : np.ndarray
            the input vector to normalize

        Returns
        -------
        ans : np.ndarray
            The normalized input vector.
        """
        return _norm_input_helper(val, self._offset_list, self._spacing_list)

    # The following (in)equality operations generate (in)equality constraints for each sub-function
    def __eq__(self, other):
        return [fn.__eq__(other) for fn in self._fn_list]

    def __ge__(self, other):
        return [fn.__ge__(other) for fn in self._fn_list]

    def __le__(self, other):
        return [fn.__le__(other) for fn in self._fn_list]

    def __gt__(self, other):
        return self.__ge__(other)

    def __lt__(self, other):
        return self.__le__(other)

    def random_input(self, rng: Optional[np.random.Generator] = None, norm: bool = True) -> List[float]:
        """Generates a random input.

        Parameters
        ----------
        rng : Optional[np.random.Generator]
            the random number generator to use.
        norm : bool
            True to return normalized random inputs. Default is True.

        Returns
        -------
        ans : List[float]
            The randomly generated input vector.
        """
        if rng is None:
            rng = np.random.default_rng()
        rv = []
        idx = 0
        bnd_list = self.input_ranges_norm if norm else self.input_ranges
        for dim_idx in range(self._fn_list[0].inner_ndim):
            if dim_idx in self._fixed_vals_list[0]:  # fixed value is used, cannot be randomized
                continue
            val = rng.uniform(*bnd_list[idx])
            if dim_idx in self._indep_idx_list:  # share the same random value
                rv += [val] * self.num_fn
                idx += self.num_fn
            else:
                rv.append(val)
                idx += 1
        return rv


class CustomVectorReduceArgMapFunction(CustomVectorArgMapFunction):
    """A CustomVectorArgMapFunction that performs vector reduction.

    Parameters
    ----------
    Refer to CustomVectorArgMapFunction for the shared parameters.

    reduce_fn : Callable
        the reduction function.
    """
    def __init__(self, *args, reduce_fn: Callable):
        super().__init__(*args)
        self._reduce_fn = reduce_fn

    def __call__(self, xi: np.ndarray, **kwargs) -> np.ndarray:
        return self._reduce_fn(super().__call__(xi, **kwargs))


def _denorm_input_elem_helper(val: Union[float, np.ndarray], idx: int,
                              offset_list: np.ndarray, spacing_list: np.ndarray) -> Union[float, np.ndarray]:
    """A helper function for de-normalizing an input element.

    Parameters
    ----------
    val : Union[float, np.ndarray]
        the element to de-normalize
    idx : int
        the index to de-normalize on
    offset_list : np.ndarray
        the offset list
    spacing_list : np.ndarray
        the spacing list

    Returns
    -------
    ans : np.ndarray
        The de-normalized input element.
    """
    return val * spacing_list[idx] + offset_list[idx]


def _norm_input_elem_helper(val: Union[float, np.ndarray], idx: int,
                            offset_list: np.ndarray, spacing_list: np.ndarray) -> Union[float, np.ndarray]:
    """A helper function for normalizing an input element.

    Parameters
    ----------
    val : Union[float, np.ndarray]
        the element to normalize
    idx : int
        the index to normalize on
    offset_list : np.ndarray
        the offset list
    spacing_list : np.ndarray
        the spacing list

    Returns
    -------
    ans : np.ndarray
        The normalized input element.
    """
    return (val - offset_list[idx]) / spacing_list[idx]


def _denorm_input_helper(val: np.ndarray, offset_list: np.ndarray, spacing_list: np.ndarray) -> np.ndarray:
    """A helper function for de-normalizing an input vector.

    Parameters
    ----------
    val : np.ndarray
        the input vector to de-normalize
    offset_list : np.ndarray
        the offset list
    spacing_list : np.ndarray
        the spacing list

    Returns
    -------
    ans : np.ndarray
        The de-normalized input vector.
    """
    return val * spacing_list + offset_list


def _norm_input_helper(val: np.ndarray, offset_list: np.ndarray, spacing_list: np.ndarray) -> np.ndarray:
    """A helper function for normalizing an input vector.

    Parameters
    ----------
    val : np.ndarray
        the input vector to normalize
    offset_list : np.ndarray
        the offset list
    spacing_list : np.ndarray
        the spacing list

    Returns
    -------
    ans : np.ndarray
        The normalized input vector.
    """
    return (val - offset_list) / spacing_list
