# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, Iterable, List, Set

import numpy as np

from bag.simulation.data import SimData, AnalysisType

from ..data.tran import EdgeType, get_first_crossings
from .base import TranTB


class DigitalTranTB(TranTB):
    """A transient testbench with digital stimuli.  All pins are connected to either 0 or 1.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_sim : float
            the total simulation time.
        t_rst : float
            the duration of reset signals.
        t_rst_rf : float
            the reset signals rise/fall time.
    pulse_list : Sequence[Mapping[str, Any]]
        Optional.  List of pulse sources.  Each dictionary has the following entries:

        pin : str
            the pin to connect to.
        tper : Union[float, str]
            period.
        tpw : Union[float, str]
            the pulse width, measures from 50% to 50%, i.e. it is tper/2 for 50% duty cycle.
        trf : Union[float, str]
            rise/fall time as defined by thres_lo and thres_hi.
        td : Union[float, str]
            Optional.  Pulse delay in addition to any reset period,  Measured from the end of
            reset period to the 50% point of the first edge.
        pos : bool
            Defaults to True.  True if this is a positive pulse (010).
        td_after_rst: bool
            Defaults to True.  True if td is measured from the end of reset period, False
            if td is measured from t=0.

    reset_list : Sequence[Tuple[str, bool]]
        Optional.  List of reset pin name and reset type tuples.  Reset type is True for
        active-high, False for active-low.
    rtol : float
        Optional.  Relative tolerance for equality checking in timing measurement.
    atol : float
        Optional.  Absolute tolerance for equality checking in timing measurement.
    thres_lo : float
        Optional.  Low threshold value for rise/fall time calculation.  Defaults to 0.1
    thres_hi : float
        Optional.  High threshold value for rise/fall time calculation.  Defaults to 0.9
    subclasses' specs dictionary must have pwr_domain, rtol, atol, thres_lo, and thres_hi.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._thres_lo: float = 0.1
        self._thres_hi: float = 0.9

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs

        self._thres_lo = specs.get('thres_lo', 0.1)
        self._thres_hi = specs.get('thres_hi', 0.9)
        thres_delay = specs.get('thres_delay', 0.5)
        if abs(thres_delay - 0.5) > 1e-4:
            raise ValueError('only thres_delay = 0.5 is supported.')

    @property
    def t_rst_end_expr(self) -> str:
        return f't_rst+t_rst_rf/{self.trf_scale:.2f}'

    @property
    def thres_lo(self) -> float:
        return self._thres_lo

    @property
    def thres_hi(self) -> float:
        return self._thres_hi

    @property
    def trf_scale(self) -> float:
        return self._thres_hi - self._thres_lo

    def get_t_rst_end(self, data: SimData) -> np.ndarray:
        t_rst = self.get_param_value('t_rst', data)
        t_rst_rf = self.get_param_value('t_rst_rf', data)
        return t_rst + t_rst_rf / self.trf_scale

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Set up PWL waveform files."""
        ans = super().pre_setup(sch_params)

        specs = self.specs
        pulse_list: Sequence[Mapping[str, Any]] = specs.get('pulse_list', [])
        reset_list: Sequence[Tuple[str, bool]] = specs.get('reset_list', [])

        src_list = ans['src_list']
        src_pins = set()
        self.get_pulse_sources(pulse_list, src_list, src_pins)
        self.get_reset_sources(reset_list, src_list, src_pins, skip_src=True)

        return ans

    def get_reset_sources(self, reset_list: Iterable[Tuple[str, bool]],
                          src_list: List[Mapping[str, Any]], src_pins: Set[str],
                          skip_src: bool = False) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        trf_scale = self.trf_scale
        for pin, active_high in reset_list:
            if pin in src_pins:
                if skip_src:
                    continue
                else:
                    raise ValueError(f'Cannot add reset source on pin {pin}, already used.')

            gnd_name, pwr_name = self.get_pin_supplies(pin, pwr_domain)
            if active_high:
                v1 = f'v_{pwr_name}'
                v2 = f'v_{gnd_name}'
            else:
                v1 = f'v_{gnd_name}'
                v2 = f'v_{pwr_name}'

            trf_str = f't_rst_rf/{trf_scale:.2f}'
            pval_dict = dict(v1=v1, v2=v2, td='t_rst', per='2*t_sim', pw='t_sim',
                             tr=trf_str, tf=trf_str)
            self._add_diff_sources(pin, [pval_dict], '', src_list, src_pins)

    def get_pulse_sources(self, pulse_list: Iterable[Mapping[str, Any]],
                          src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        specs = self.specs
        pwr_domain: Mapping[str, Tuple[str, str]] = specs['pwr_domain']
        skip_src: bool = specs.get('skip_src', False)

        trf_scale = self.trf_scale
        td_rst = f't_rst+(t_rst_rf/{trf_scale:.2f})'
        for pulse_params in pulse_list:
            pin: str = pulse_params['pin']
            rs: Union[float, str] = pulse_params.get('rs', '')
            vadd_list: Optional[Sequence[Mapping[str, Any]]] = pulse_params.get('vadd_list', None)

            if pin in src_pins:
                if skip_src:
                    continue
                else:
                    raise ValueError(f'Cannot add pulse source on pin {pin}, already used.')

            if not vadd_list:
                vadd_list = [pulse_params]

            gnd_name, pwr_name = self.get_pin_supplies(pin, pwr_domain)
            ptable_list = []
            for table in vadd_list:
                tper: Union[float, str] = table['tper']
                tpw: Union[float, str] = table['tpw']
                trf: Union[float, str] = table['trf']
                td: Union[float, str] = table.get('td', '')
                pos: bool = table.get('pos', True)
                td_after_rst: bool = table.get('td_after_rst', True)
                extra: Mapping[str, Union[float, str]] = table.get('extra', {})

                if pos:
                    v1 = f'v_{gnd_name}'
                    v2 = f'v_{pwr_name}'
                else:
                    v1 = f'v_{pwr_name}'
                    v2 = f'v_{gnd_name}'

                if isinstance(trf, float):
                    trf /= trf_scale
                    trf2 = self.get_sim_param_string(trf / 2)
                    trf = self.get_sim_param_string(trf)
                else:
                    trf2 = f'({trf})/{2 * trf_scale:.2f}'
                    trf = f'({trf})/{trf_scale:.2f}'

                if not td:
                    td = td_rst if td_after_rst else '0'
                else:
                    td = self.get_sim_param_string(td)
                    if td_after_rst:
                        td = f'{td_rst}+{td}-{trf2}'
                    else:
                        td = f'{td}-{trf2}'

                tpw = self.get_sim_param_string(tpw)
                ptable_list.append(dict(v1=v1, v2=v2, td=td, per=tper, pw=f'{tpw}-{trf}',
                                        tr=trf, tf=trf, **extra))

            self._add_diff_sources(pin, ptable_list, rs, src_list, src_pins)

    def calc_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
                   t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        out_vec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_out = (out_1 - out_0) * thres_delay + out_0
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        return out_c

    def calc_delay(self, data: SimData, in_name: str, out_name: str, in_edge: EdgeType,
                   out_edge: EdgeType, t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        in_0, in_1 = self.get_pin_supply_values(in_name, data)
        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        in_vec = data[in_name]
        out_vec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_in = (in_1 - in_0) * thres_delay + in_0
        vth_out = (out_1 - out_0) * thres_delay + out_0
        in_c = get_first_crossings(tvec, in_vec, vth_in, etype=in_edge, start=t_start, stop=t_stop,
                                   rtol=rtol, atol=atol)
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        out_c -= in_c
        return out_c

    def calc_trf(self, data: SimData, out_name: str, out_rise: bool, allow_inf: bool = False,
                 t_start: Union[np.ndarray, float, str] = 0,
                 t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        specs = self.specs
        logger = self.logger
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vdiff = out_1 - out_0
        vth_0 = out_0 + self._thres_lo * vdiff
        vth_1 = out_0 + self._thres_hi * vdiff
        if out_rise:
            edge = EdgeType.RISE
            t0 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
        else:
            edge = EdgeType.FALL
            t0 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)

        has_nan = np.isnan(t0).any() or np.isnan(t1).any()
        has_inf = np.isinf(t0).any() or np.isinf(t1).any()
        if has_nan or (has_inf and not allow_inf):
            logger.warn(f'Got invalid value(s) in computing {edge.name} time of pin {out_name}.\n'
                        f't0:\n{t0}\nt1:\n{t1}')
            t1.fill(np.inf)
        else:
            t1 -= t0

        return t1

    def _add_diff_sources(self, pin: str, ptable_list: Sequence[Mapping[str, Any]],
                          rs: Union[float, str], src_list: List[Mapping[str, Any]],
                          src_pins: Set[str]) -> None:
        pos_pins, neg_pins = self.get_diff_groups(pin)
        self._add_diff_sources_helper(pos_pins, ptable_list, rs, src_list, src_pins)
        if neg_pins:
            ntable_list = []
            for ptable in ptable_list:
                ntable = dict(**ptable)
                ntable['v1'] = ptable['v2']
                ntable['v2'] = ptable['v1']
                ntable_list.append(ntable)

            self._add_diff_sources_helper(neg_pins, ntable_list, rs, src_list, src_pins)

    def _add_diff_sources_helper(self, pin_list: Sequence[str],
                                 table_list: Sequence[Mapping[str, Any]],
                                 rs: Union[float, str], src_list: List[Mapping[str, Any]],
                                 src_pins: Set[str]) -> None:
        num_pulses = len(table_list)
        for pin_name in pin_list:
            if pin_name in src_pins:
                raise ValueError(f'Cannot add pulse source on pin {pin_name}, '
                                 f'already used.')
            if rs:
                pulse_pin = self.get_r_src_pin(pin_name)
                src_list.append(dict(type='res', lib='analogLib', value=rs,
                                     conns=dict(PLUS=pin_name, MINUS=pulse_pin)))
            else:
                pulse_pin = pin_name

            bot_pin = 'VSS'
            for idx, table in enumerate(table_list):
                top_pin = pulse_pin if idx == num_pulses - 1 else f'{pin_name}_vadd{idx}_'
                src_list.append(dict(type='vpulse', lib='analogLib', value=table,
                                     conns=dict(PLUS=top_pin, MINUS=bot_pin)))
                bot_pin = top_pin
            src_pins.add(pin_name)
