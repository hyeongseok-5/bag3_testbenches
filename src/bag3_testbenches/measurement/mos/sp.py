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

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, cast
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData
from bag.concurrent.util import GatherHelper

from ..sp.base import SPTB
from ..data.tran import get_first_crossings, EdgeType


class MOSSPMeas(MeasurementManager):
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        helper = GatherHelper()
        sim_envs = self.specs['sim_envs']
        for sim_env in sim_envs:
            helper.append(self.async_meas_pvt(name, sim_dir / sim_env, sim_db, dut, harnesses, sim_env))

        meas_results = await helper.gather_err()
        results = {}
        for idx, sim_env in enumerate(sim_envs):
            results[sim_env] = meas_results[idx]
        plot_results(results)
        return results

    async def async_meas_pvt(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                             harnesses: Optional[Sequence[DesignInstance]], pvt: str) -> Mapping[str, Any]:
        # add port on G and D using dcblock
        load_list = [dict(pin='gport', nin='s', type='port', value={'r': 50}, name='PORTG'),
                     dict(pin='gport', nin='g', type='dcblock', value=''),
                     dict(pin='dport', nin='s', type='port', value={'r': 50}, name='PORTD'),
                     dict(pin='dport', nin='d', type='dcblock', value=''),
                     ]

        # add DC bias using dcfeed
        load_list.extend([dict(pin='gbias', nin='s', type='vdc', value='vgs'),
                          dict(pin='gbias', nin='g', type='dcfeed', value=''),
                          dict(pin='dbias', nin='s', type='vdc', value='vds'),
                          dict(pin='dbias', nin='d', type='dcfeed', value=''),
                          ])

        # set vgs and vds sweep info
        vds_val = self.specs['vds_val']
        vgs_val = self.specs['vgs_val']

        tbm_specs = dict(
            **self.specs['tbm_specs'],
            load_list=load_list,
            sim_envs=[pvt],
            swp_info=[('vds', dict(type='LIST', values=vds_val)),
                      ('vgs', dict(type='LIST', values=vgs_val))],
            dut_pins=dut.pin_names,
            param_type='Y',
            ports=['PORTG', 'PORTD'],
        )
        tbm = cast(SPTB, self.make_tbm(SPTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, {'dut_conns': {'b': 's', 'd': 'd',
                                                                                                  'g': 'g', 's': 's'}},
                                                          harnesses=harnesses)
        results = calc_ft_fmax(sim_results.data)
        return results


def calc_ft_fmax(sim_data: SimData) -> Mapping[str, Any]:
    freq = sim_data['freq']
    y11 = sim_data['y11']
    y12 = sim_data['y12']
    y21 = sim_data['y21']
    y22 = sim_data['y22']

    # calculate fT: freq where h21 reaches 1
    h21 = y21 / y11
    ft = get_first_crossings(freq, np.abs(h21), 1, etype=EdgeType.FALL)

    # calculate fmax: freq where unilateral gain U reaches 1
    _num = np.abs(y21 - y12) ** 2
    _den = 4 * (y11.real * y22.real - y12.real * y21.real)
    ug = _num / _den
    fmax = get_first_crossings(freq, ug, 1, etype=EdgeType.FALL)

    return dict(ft=ft[0], fmax=fmax[0], vgs=sim_data['vgs'], vds=sim_data['vds'])


def plot_results(results: Mapping[str, Any]) -> None:
    vgs = None
    vds = None
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.set(xlabel='vgs (V)', ylabel='fT (GHz)', title='fT vs vgs')
    ax1.set(xlabel='vgs (V)', ylabel='fmax (GHz)', title='fmax vs vgs')
    for sim_env, _results in results.items():
        if vgs is None:
            vgs = _results['vgs']
            vds = _results['vds']
        for idx, _vds in enumerate(vds):
            ax0.plot(vgs, _results['ft'][idx] * 1e-9, label=f'vds={_vds}, {sim_env}')
            ax1.plot(vgs, _results['fmax'][idx] * 1e-9, label=f'vds={_vds}, {sim_env}')
    ax0.legend()
    ax1.legend()
    plt.tight_layout()
    plt.show()
