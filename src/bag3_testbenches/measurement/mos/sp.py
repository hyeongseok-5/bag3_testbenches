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

from ..sp.base import SPTB
from ..data.tran import get_first_crossings, EdgeType


class MOSSPMeas(MeasurementManager):
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
        return results

    async def async_meas_pvt(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                             harnesses: Optional[Sequence[DesignInstance]], pvt: str) -> Mapping[str, Any]:
        # add port on G and D
        load_list = [dict(conns={'PLUS': 'g', 'MINUS': 's'}, type='port', value={'r': 50, 'vdc': 'vgs'}, name='PORTG'),
                     dict(conns={'PLUS': 'd', 'MINUS': 's'}, type='port', value={'r': 50, 'vdc': 'vds'}, name='PORTD')]

        # set vgs and vds sweep info
        vds_val = self.specs['vds_val']
        vgs_val = self.specs['vgs_val']

        tbm_specs = dict(
            **self.specs['tbm_specs'],
            load_list=load_list,
            sim_envs=[pvt],
            swp_info=[('vgs', dict(type='LIST', values=vgs_val)),
                      ('vds', dict(type='LIST', values=vds_val))],
            dut_pins=dut.pin_names,
            param_type='Y',
            ports=['PORTG', 'PORTD'],
        )
        tbm = cast(SPTB, self.make_tbm(SPTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, {'dut_conns': {'b': 's', 'd': 'd',
                                                                                                  'g': 'g', 's': 's'}},
                                                          harnesses=harnesses)
        results = self.post_process(sim_results.data)
        return results

    @classmethod
    def post_process(cls, sim_data: SimData) -> Mapping[str, Any]:
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

        return dict(ft=ft, fmax=fmax, vgs=sim_data['vgs'], vds=sim_data['vds'])
