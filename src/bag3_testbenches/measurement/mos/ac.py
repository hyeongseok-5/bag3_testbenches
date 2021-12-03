from __future__ import annotations

from typing import Any, Mapping, Optional, Union, Sequence, Tuple, cast
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimData
from bag.concurrent.util import GatherHelper

from ..ac.base import ACTB
from ..data.tran import get_first_crossings, EdgeType


class MOSACMeas(MeasurementManager):
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo,
                     harnesses: Optional[Sequence[DesignInstance]] = None
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance,
                   harnesses: Optional[Sequence[DesignInstance]] = None) -> Tuple[bool, MeasInfo]:
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise NotImplementedError

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
        # add AC sources on G and D
        load_list = [dict(pin='g', nin='s', type='vdc', value={'vdc': 'vgs', 'acm': 1}, name='VDCG'),
                     dict(pin='d', nin='s', type='vdc', value={'vdc': 'vds'}, name='VDCD')]

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
            save_outputs=['VDCG:p', 'VDCD:p'],
        )
        tbm = cast(ACTB, self.make_tbm(ACTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, {'dut_conns': {'b': 's', 'd': 'd',
                                                                                                  'g': 'g', 's': 's'}},
                                                          harnesses=harnesses)
        results = calc_ft_fmax(sim_results.data)
        return results


def calc_ft_fmax(sim_data: SimData) -> Mapping[str, Any]:
    freq = sim_data['freq']
    i_g = sim_data['VDCG:p']
    i_d = sim_data['VDCD:p']

    # calculate fT: freq where h21 reaches 1
    h21 = i_d / i_g
    ft = get_first_crossings(freq, np.abs(h21), 1, etype=EdgeType.FALL)

    return dict(ft=ft[0], vgs=sim_data['vgs'], vds=sim_data['vds'])


def plot_results(results: Mapping[str, Any]) -> None:
    vgs = None
    vds = None
    fig, (ax0) = plt.subplots(1, 1)
    ax0.set(xlabel='vgs (V)', ylabel='fT (GHz)', title='fT vs vgs')
    for sim_env, _results in results.items():
        if vgs is None:
            vgs = _results['vgs']
            vds = _results['vds']
        for idx, _vds in enumerate(vds):
            ax0.plot(vgs, _results['ft'][idx] * 1e-9, label=f'vds={_vds}, {sim_env}')
    ax0.legend()
    plt.tight_layout()
    plt.show()
