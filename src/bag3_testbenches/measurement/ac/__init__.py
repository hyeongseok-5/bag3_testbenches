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

from typing import Any, Sequence, Optional, Mapping, Type, List, Set, Union, Tuple, Iterable

from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict
from bag.design.module import Module

from bag3_liberty.data import parse_cdba_name, BusRange

from ...schematic.digital_tb_tran import bag3_testbenches__digital_tb_tran


class ACTB(TestbenchManager):
    """This class provide utility methods useful for all ac simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sweep_var : str
        The variable to be swept.
    sweep_options : Mapping[str, Any]
        Dictionary with following entries :
            type : str
                type of DC sweep (LINEAR / LOG)
            start : Union[str, float]
                initial value of sweep_var
            stop : Union[str, float]
                final value of sweep_var
            num : int
                number of sweep data points
    dut_pins : Sequence[str]
        list of DUT pins.
    src_list : Sequence[Mapping[str, Any]]
        list of DC sources.
    load_list : Sequence[Mapping[str, Any]]
        Optional.  List of loads.  Each dictionary has the following entries:

        pin: str
            the pin to connect to.
        type : str
            the load device type.
        value : Union[float, str]
            the load parameter value.
    pwr_domain : Mapping[str, Tuple[str, str]]
        Dictionary from individual pin names or base names to (ground, power) pin name tuple.
    sup_values : Mapping[str, Union[float, Mapping[str, float]]]
        Dictionary from supply pin name to voltage values.

    subclasses can define the following optional entries:

    save_outputs : Sequence[str]
        Optional.  list of nets to save in simulation data file.
    ac_options : Mapping[str, Any]
        Optional.  ac simulation options dictionary.
    """

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__digital_tb_tran

    @classmethod
    def sup_var_name(cls, sup_pin: str) -> str:
        return f'v_{sup_pin}'

    @classmethod
    def get_pin_supplies(cls, pin_name: str, pwr_domain: Mapping[str, Tuple[str, str]]
                         ) -> Tuple[str, str]:
        ans = pwr_domain.get(pin_name, None)
        if ans is None:
            basename, _ = parse_cdba_name(pin_name)
            return pwr_domain[basename]
        return ans

    def get_bias_sources(self, sup_values: Mapping[str, Union[float, Mapping[str, float]]],
                         src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        """Save bias sources and pins into src_list and src_pins.

        Side effect: will add voltage variables in self.sim_params.
        """
        sim_params = self.sim_params
        env_params = self.env_params
        for sup_pin, sup_val in sup_values.items():
            if sup_pin in src_pins:
                raise ValueError(f'Cannot add bias source on pin {sup_pin}, already used.')

            var_name = self.sup_var_name(sup_pin)
            if sup_pin == 'VSS':
                if sup_val != 0:
                    raise ValueError('VSS must be 0 volts.')
            else:
                src_list.append(dict(type='vdc', lib='analogLib', value=var_name,
                                     conns=dict(PLUS=sup_pin, MINUS='VSS')))
                src_pins.add(sup_pin)
            if isinstance(sup_val, float) or isinstance(sup_val, int):
                sim_params[var_name] = float(sup_val)
            else:
                env_params[var_name] = dict(**sup_val)

    def get_dut_conns(self, dut_pins: Iterable[str], src_pins: Set[str],
                      pin_values: Mapping[str, int]) -> Mapping[str, str]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        ans = {}
        for pin_name in dut_pins:
            pin_val: Optional[int] = pin_values.get(pin_name, None)
            basename, bus_range = parse_cdba_name(pin_name)
            if bus_range is None:
                # scalar pins
                if pin_name in src_pins or pin_val is None:
                    ans[pin_name] = pin_name
                else:
                    ans[pin_name] = self.get_pin_supplies(pin_name, pwr_domain)[pin_val]
            else:
                # bus pins
                if pin_val is None:
                    # no bias values specified
                    ans[pin_name] = pin_name
                else:
                    nlen = len(bus_range)
                    bin_str = bin(pin_val)[2:].zfill(nlen)
                    ans[pin_name] = self._bin_str_to_net(basename, bus_range, bin_str, pwr_domain,
                                                         src_pins)

        return ans

    def _bin_str_to_net(self, basename: str, bus_range: BusRange, bin_str: str,
                        pwr_domain: Mapping[str, Tuple[str, str]], src_pins: Set[str]) -> str:
        last_pin = ''
        cur_cnt = 0
        net_list = []
        for bus_idx, char in zip(bus_range, bin_str):
            cur_pin = f'{basename}<{bus_idx}>'
            if cur_pin not in src_pins:
                cur_pin = self.get_pin_supplies(cur_pin, pwr_domain)[int(char == '1')]

            if cur_pin == last_pin:
                cur_cnt += 1
            else:
                if last_pin:
                    net_list.append(last_pin if cur_cnt == 1 else f'<*{cur_cnt}>{last_pin}')
                last_pin = cur_pin
                cur_cnt = 1

        if last_pin:
            net_list.append(last_pin if cur_cnt == 1 else f'<*{cur_cnt}>{last_pin}')
        return ','.join(net_list)

    def get_loads(self, load_list: Iterable[Mapping[str, Any]],
                  src_load_list: List[Mapping[str, Any]]) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        for params in load_list:
            pin: str = params['pin']
            value: Union[float, str] = params['value']
            dev_type: str = params['type']
            gnd_name = self.get_pin_supplies(pin, pwr_domain)[0]
            src_load_list.append(dict(type=dev_type, lib='analogLib', value=value,
                                      conns=dict(PLUS=pin, MINUS=gnd_name)))

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if sch_params is None:
            return None

        specs = self.specs
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']
        dut_pins: Sequence[str] = specs['dut_pins']
        pin_values: Mapping[str, int] = specs['pin_values']
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        src_list: List[Mapping[str, Any]] = specs.get('src_list', [])
        src_pins = set()
        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_loads(load_list, src_list)

        dut_conns = self.get_dut_conns(dut_pins, src_pins, pin_values)
        return dict(
            dut_lib=sch_params.get('dut_lib', ''),
            dut_cell=sch_params.get('dut_cell', ''),
            dut_params=sch_params.get('dut_params', None),
            dut_conns=sch_params.get('dut_conns', dut_conns),
            vbias_list=[],
            src_list=src_list,
        )

    def get_netlist_info(self) -> SimNetlistInfo:
        sweep_var: str = self.specs.get('sweep_var', 'freq')
        sweep_options: Mapping[str, Any] = self.specs['sweep_options']
        ac_options: Mapping[str, Any] = self.specs.get('ac_options', {})
        save_outputs: Sequence[str] = self.specs.get('save_outputs', ['plus', 'minus'])
        ac_dict = dict(type='AC',
                       param=sweep_var,
                       sweep=sweep_options,
                       options=ac_options,
                       save_outputs=save_outputs,
                       )

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [ac_dict]
        return netlist_info_from_dict(sim_setup)
