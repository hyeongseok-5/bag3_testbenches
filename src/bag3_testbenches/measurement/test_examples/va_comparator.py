from typing import Any, Mapping, Optional, Union, Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.data import SimData

from ..tran.digital import DigitalTranTB


class VerilogAComparatorMeas(MeasurementManager):
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo,
                     harnesses: Optional[Sequence[DesignInstance]] = None):
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance,
                   harnesses: Optional[Sequence[DesignInstance]] = None):
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]):
        raise NotImplementedError

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        # create inputs
        pulse_list = [dict(pin='v_ref', tper='t_per', tpw='t_per/2', trf='t_rf')]
        load_list = [dict(pin='v_in', type='vdc', value='v_ref')]
        save_outputs = ['v_in', 'v_ref']

        # create veriloga instances
        va_list: Sequence[Mapping[str, Any]] = self.specs['va_list']
        load_list.extend(va_list)
        va_cvinfo: Sequence[str] = self.specs['va_cvinfo']

        extra_outputs: Sequence[str] = self.specs['save_outputs']
        save_outputs.extend(extra_outputs)

        tbm_specs = dict(**self.specs['tbm_specs'])
        tbm_specs.update(dict(
            pulse_list=pulse_list,
            load_list=load_list,
            sim_envs=self.specs['sim_envs'],
            save_outputs=save_outputs,
            dut_pins=[],
        ))
        tbm = self.make_tbm(DigitalTranTB, tbm_specs)
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, {'va_cvinfo': va_cvinfo})
        self.plot_data(sim_results.data, save_outputs)
        return {}

    @staticmethod
    def plot_data(sim_data: SimData, save_outputs: Sequence[str]) -> None:
        time = sim_data['time']
        plt.figure()
        for sig_name in save_outputs:
            sig = np.squeeze(sim_data[sig_name])
            plt.plot(time, sig, label=sig_name)
        plt.grid()
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('voltage (V)')
        plt.show()
