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

from typing import Type, Union, Optional, Any, cast, Sequence, Mapping, Tuple
from pathlib import Path
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.design.module import Module
from bag.concurrent.util import GatherHelper
from bag.math import float_to_si_string

from ...schematic.char_tb_ac import bag3_testbenches__char_tb_ac


class CharACTB(TestbenchManager):
    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__char_tb_ac

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


class CharACMeas(MeasurementManager):
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo):
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance):
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]):
        raise NotImplementedError

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Mapping[int, Mapping[str, Any]]:
        helper = GatherHelper()
        ibias_list = self.specs.get('ibias_list', [0])
        for ibias in ibias_list:
            for idx in range(2):
                helper.append(self.async_meas_case(name, sim_dir, sim_db, dut, idx, ibias))

        meas_results = await helper.gather_err()
        passive_type: str = self.specs['passive_type']
        ans = {}
        for idx, ibias in enumerate(ibias_list):
            midx = idx * 2
            _results = meas_results[midx:(midx + 2)]
            _ans = compute_passives(_results, passive_type)
            ans[idx] = dict(
                **_ans,
                ibias=ibias,
            )
        return ans

    async def async_meas_case(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                              case_idx: int, ibias: float = 0.0) -> Mapping[str, Any]:
        if case_idx == 0:
            sup_conns = [('PLUS', 'plus'), ('MINUS', 'minus')]
        elif case_idx == 1:
            sup_conns = [('PLUS', 'plus'), ('MINUS', 'VSS')]
        else:
            raise ValueError(f'Invalid case_idx={case_idx}')

        tbm_specs = dict(
            **self.specs['tbm_specs']['ac_meas'],
            sim_envs=self.specs['sim_envs'],
        )
        tbm_specs['sim_params']['idc'] = ibias
        tbm = cast(CharACTB, self.make_tbm(CharACTB, tbm_specs))
        tbm_name = f'{name}_{case_idx}_{float_to_si_string(ibias)}'
        sim_dir = sim_dir / tbm_name

        passive_type: str = self.specs['passive_type']
        if passive_type == 'ind':
            sp_file = Path(self.specs['sp_file'])
            ind_sp = f'{sp_file.stem}.s1p'
            sim_dir.mkdir(parents=True, exist_ok=True)
            copy(sp_file, sim_dir / ind_sp)
        else:
            ind_sp = ''

        tb_params = dict(
            extracted=self.specs['tbm_specs'].get('extracted', True),
            sup_conns=sup_conns,
            passive_type=passive_type,
            ind_sp=ind_sp,
        )
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir, dut, tbm,
                                                          tb_params=tb_params)
        data = sim_results.data
        return dict(freq=data['freq'], plus=np.squeeze(data['plus']), minus=np.squeeze(data['minus']))


def estimate_cap(freq: np.ndarray, zc: np.ndarray) -> float:
    """assume zc = 1 / jwC"""
    fit = np.polyfit(2 * np.pi * freq, - 1 / np.imag(zc), 1)
    return fit[0]


def estimate_ind(freq: np.ndarray, zc: np.ndarray) -> Mapping[str, float]:
    """assume res and ind in series; cap in parallel"""
    w = 2 * np.pi * freq

    # find SRF: max freq where zc.imag goes from positive to negative
    loc1 = np.where(zc.imag > 0)[0][-1]
    # Linearly interpolate between loc and (loc + 1)
    w1, z1 = w[loc1], zc.imag[loc1]
    w2, z2 = w[loc1 + 1], zc.imag[loc1 + 1]
    w0 = w2 - z2 * (w2 - w1) / (z2 - z1)

    # upto 0.1 * SRF, assume zc is just R + jwL
    idx0 = np.where(w < 0.1 * w0)[0][-1]
    res = np.mean(zc.real[:idx0])
    _fit = np.polyfit(w[:idx0], zc.imag[:idx0], 1)
    ind = _fit[0]

    cap = 1 / (w0 * w0 * ind)
    ans = dict(ind=ind, res=res, cap=cap)

    # --- Debug plots --- #
    # plt.semilogx(freq[:idx0], zc.imag[:idx0], label='Measured')
    # plt.semilogx(freq[:idx0], w[:idx0] * ind, label='Estimated')
    # plt.xlabel('Frequency (in Hz)')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    return ans


def estimate_esd(freq: np.ndarray, zc: np.ndarray) -> Tuple[float, float]:
    """assume zc = R / (1 + jwCR); returns C, R"""
    yc = 1 / zc  # yc = (1/R) + jwC
    fit = np.polyfit(2 * np.pi * freq, np.imag(yc), 1)
    cap: float = fit[0]
    res: float = 1 / np.mean(yc.real)
    return cap, res


def compute_passives(meas_results: Sequence[Mapping[str, Any]], passive_type: str) -> Mapping[str, Any]:
    freq0 = meas_results[0]['freq']
    freq1 = meas_results[1]['freq']
    assert np.isclose(freq0, freq1).all()

    # vm0 = (zc * zpm) / (zc + zpp + zpm)
    # vp0 = - (zc * zpp) / (zc + zpp + zpm)
    vm0 = meas_results[0]['minus']
    vp0 = meas_results[0]['plus']

    # vm1 = - (zpp * zpm) / (zc + zpp + zpm)
    # vp1 = - ((zc + zpm) * zpp) / (zc + zpp + zpm)
    vm1 = meas_results[1]['minus']
    vp1 = meas_results[1]['plus']

    # --- Find zc, zpp, zpm using vm0, vp0, vm1 --- #
    # - vp0 / vm0 = zpp / zpm = const_a  ==> zpp = const_a * zpm
    const_a = - vp0 / vm0
    # vp0 / vm1 = zc / zpm = const_b  ==> zc = const_b * zpm
    const_b = vp0 / vm1

    # vp0 = - (const_b * const_a * zpm) / (const_b + const_a + 1)
    zpm = - vp0 * (const_b + const_a + 1) / (const_b * const_a)
    zpp = const_a * zpm
    zc = const_b * zpm

    # --- Verify vp1 is consistent --- #
    vp1_calc = - ((zc + zpm) * zpp) / (zc + zpp + zpm)
    if not np.isclose(vp1, vp1_calc, rtol=1e-3).all():
        plt.loglog(freq0, np.abs(vp1), label='measured')
        plt.loglog(freq0, np.abs(vp1_calc), 'g--', label='calculated')
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    results = dict(
        cpp=estimate_cap(freq0, zpp),
        cpm=estimate_cap(freq0, zpm),
    )
    if passive_type == 'cap':
        results['cc'] = estimate_cap(freq0, zc)
        results['r_series'] = np.mean(zc).real
    elif passive_type == 'res':
        results['res'] = np.mean(zc).real
    elif passive_type == 'esd':
        results['cc'], results['res'] = estimate_esd(freq0, zc)
    elif passive_type == 'ind':
        ind_values = estimate_ind(freq0, zc)
        results.update(ind_values)
        # --- Debug plots --- #
        warr = 2 * np.pi * freq0
        z_est = 1 / (1 / (ind_values['res'] + 1j * warr * ind_values['ind']) + 1j * warr * ind_values['cap'])
        plt.semilogx(freq0, zc.imag, label='Measured')
        plt.semilogx(freq0, z_est.imag, label='Estimated')
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    else:
        raise ValueError(f'Unknown passive_type={passive_type}. Use "cap" or "res" or "esd" or "ind".')
    return results
