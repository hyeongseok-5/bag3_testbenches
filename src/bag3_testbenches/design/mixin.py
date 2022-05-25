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

"""This module defines some utility methods and classes to support the interface between designer and
generator parameters."""

import abc
import functools
from typing import Optional, Mapping, Any, Union, Dict, Sequence, Type, List

from bag.util.immutable import Param


def get_dut_param_value(name: str, dsn_params: Mapping[str, Any], gen_specs: Optional[Mapping[str, Any]] = None,
                        gen_key: Optional[Sequence] = None, default_val: Optional[Any] = None,
                        dtype: Optional[Type] = None):
    """Returns a DUT parameter value by performing lookup in different tables. The lookup order is as follows:
    1. Design parameters (by parameter name)
    2. Default generator specs (by gen specs-specific nested key)
    3. Default value

    Parameter
    ---------
    name : str
        The parameter name.

    dsn_params : Mapping[str, Any]
        The mapping of design parameters.

    gen_specs : Optional[Mapping[str, Any]]
        The optional default generator specs.

    gen_key : Optional[Sequence]
        The optional nested lookup key for generator specs.

    default_val : Optional[Any]
        The default value.

    dtype : Optional[Type]
        The optional value data type, used for casting.

    Returns
    -------
    param_val : Any
        The parameter value at params[nested_key[0]][nested_key[1]]...[nested_key[-1]].
    """
    if gen_specs:
        if name in dsn_params:
            val = dsn_params[name]
        else:
            gen_key = gen_key or []
            try:
                val = get_nested_param_value(gen_specs, gen_key)
            except KeyError:
                val = default_val
    else:
        val = dsn_params[name]
    return val if dtype is None else dtype(val)


def get_nested_param_value(params: Mapping, nested_key: Sequence) -> Any:
    """Returns a parameter value from a nested parameter dictionary.

    Parameter
    ---------
    params : Mapping
        The mapping of parameters, of any nesting depth.

    nested_key : Sequence
        The nested key, specified as a sequence of keys for to access a sub-section of params.

    Returns
    -------
    param_val : Any
        The parameter value at params[nested_key[0]][nested_key[1]]...[nested_key[-1]].
    """
    return functools.reduce(lambda sub_dict, k: sub_dict[k], [params] + list(nested_key))


def set_nested_param_value(params: Dict, nested_key: Sequence, val: Any):
    """Sets a parameter value in a nested parameter dictionary.

    Parameter
    ---------
    params : Dict
        The mapping of parameters, of any nesting depth.

    nested_key : Sequence
        The nested key, specified as a sequence of keys for to access a sub-section of params.

    val : Any
        The parameter value to set at params[nested_key[0]][nested_key[1]]...[nested_key[-1]].
    """
    if not nested_key:
        raise ValueError("Empty key")
    sub_params = functools.reduce(lambda sub_dict, k: sub_dict[k], [params] + list(nested_key[:-1]))
    sub_params[nested_key[-1]] = val


class DesignerMixin:
    """A generic mixin class for design scripts that contains some utility methods.
    Most notably, it contains a skeleton for mapping design parameter names to generator spec keys via lookup tables.

    Subclasses of this mixin should specify the following class variables:

    dsn_to_gen_spec_map_sch : Mapping[str, Sequence[Union[str, int]]]
        The mapping of design parameter name to schematic generator parameter nested key.
        If left unspecified, schematic is assumed to be unsupported.

    dsn_to_gen_spec_map_lay : Mapping[str, Sequence[Union[str, int]]]
        The mapping of design parameter name to layout generator parameter nested key.
        If left unspecified, layout is assumed to be unsupported.

    dsn_to_gen_val_map_lay : Mapping[str, Any]
        The mapping of design parameter name to layout generator's default/assumed value.
        In some cases, layout generators may not implement a corresponding schematic generator parameter and instead
        assume or force some value. In such a case, the design parameter should be set to a fixed value from this
        mapping.
    """
    dsn_to_gen_spec_map_sch: Mapping[str, Sequence[Union[str, int]]]
    dsn_to_gen_spec_map_lay: Mapping[str, Sequence[Union[str, int]]]
    dsn_to_gen_val_map_lay: Mapping[str, Any] = {}

    @classmethod
    @abc.abstractmethod
    def get_dut_gen_specs(cls, is_lay: bool, base_gen_specs: Param, dsn_params: Mapping[str, Any]) \
            -> Union[Param, Dict[str, Any]]:
        """Returns the updated generator specs with some design variables.

        Parameters
        ----------
        is_lay : bool
            True if DUT is layout, False if schematic.

        base_gen_specs : Param
            The base/default generator specs.

        dsn_params : Mapping[str, Any]
            The design variables.

        Returns
        -------
        gen_specs : Union[Param, Dict[str, Any]]
            The updated generator specs.
        """
        raise NotImplementedError

    @classmethod
    def get_dsn_to_gen_spec_map(cls, is_lay: bool) -> Mapping[str, Sequence[Union[str, int]]]:
        """Return the design parameter to generator specs key lookup table."""
        return cls.dsn_to_gen_spec_map_lay if is_lay else cls.dsn_to_gen_spec_map_sch

    @classmethod
    def get_dsn_to_gen_val_map(cls, is_lay: bool) -> Mapping[str, Any]:
        """Return the design parameter to generator parameter value lookup table."""
        return cls.dsn_to_gen_val_map_lay if is_lay else {}

    @classmethod
    def get_dsn_to_gen_var_list(cls, is_lay: bool) -> List[str]:
        """Return the list of design variables that are supported in the designer to generator mapping."""
        return list(cls.get_dsn_to_gen_spec_map(is_lay)) + list(cls.get_dsn_to_gen_val_map(is_lay))

    @classmethod
    def get_dut_param_value(cls, name: str, dsn_params: Mapping[str, Any], is_lay: bool,
                            gen_specs: Optional[Mapping[str, Any]] = None, default_val: Optional[Any] = None,
                            dtype: Optional = None) -> Any:
        """Returns a DUT parameter value by performing lookup in different tables. A wrapper around the module function
        of the same name.

        Parameter
        ---------
        name : str
            The parameter name.

        dsn_params : Mapping[str, Any]
            The mapping of design parameters.

        is_lay : bool
            True if the DUT generator is a layout generator, False if schematic generator.

        gen_specs : Optional[Mapping[str, Any]]
            The optional default generator specs.

        default_val : Optional[Any]
            The default value.

        dtype : Optional[Type]
            The optional value data type, used for casting.

        Returns
        -------
        param_val : Any
            The DUT parameter value.
        """
        dsn_to_gen_val_map = cls.get_dsn_to_gen_val_map(is_lay)
        if name in dsn_to_gen_val_map:
            return dsn_to_gen_val_map[name]
        gen_key = cls.get_dsn_to_gen_spec_map(is_lay)[name]
        return get_dut_param_value(name, dsn_params, gen_specs, gen_key, default_val, dtype)

    @classmethod
    def set_dut_gen_param_value(cls, name: str, val: Any, is_lay: bool, gen_specs: Dict[str, Any]) -> Any:
        """Sets a parameter value in the generator specs.

        Parameter
        ---------
        name : str
            The parameter name.

        val : Any
            The parameter value.

        is_lay : bool
            True if the DUT generator is a layout generator, False if schematic generator.

        gen_specs : Dict[str, Any]
            The generator specs to update.
        """
        gen_key = cls.get_dsn_to_gen_spec_map(is_lay)[name]
        set_nested_param_value(gen_specs, gen_key, val)

    @staticmethod
    def calc_in_sizing(load: Optional[int], fanout: float, size_step: int = 1, size_min: int = 1,
                       is_optional: bool = False) -> Optional[int]:
        """Computes input sizing based on load sizing and desired fanout.

        Parameters
        ----------
        load : Optional[int]
            The load sizing.

        fanout : float
            The desired fanout.

        size_step : int
            The input sizing quantization.

        size_min : int
            The minimum input size.

        is_optional : bool
            True if this computation is optional (in which case if load is None, input sizing is also None).
            If False, error if load is None. Defaults to False.

        Returns
        -------
        size_in : Optional[int]
            The input sizing.
        """
        if load is None:
            if is_optional:
                return None
            else:
                raise ValueError(f"load ({load}) must be defined")
        size_in = int(size_step * round(load / (fanout * size_step)))
        return max(size_in, size_min)

    @classmethod
    def calc_in_sizing_chain(cls, load: Optional[int], num_stages: int, fanout: float, **kwargs) -> List[Optional[int]]:
        """Computes sizing of the input chain.

        Parameters
        ----------
        load : Optional[int]
            The load sizing.

        num_stages : int
            The number of stages.

        fanout : float
            The total fanout.

        Returns
        -------
        sizes : List[Optional[int]]
            The sizing of the input chain.
        """
        sizes = []
        cur_load = load
        for i in range(num_stages):
            cur_load = cls.calc_in_sizing(cur_load, fanout, **kwargs)
            sizes.append(cur_load)
        return sizes[::-1]
