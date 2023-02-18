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

from typing import Any, Mapping, Optional, Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData

from ..tran.digital import DigitalTranTB


class VerilogAComparatorMeas(MeasurementManager):
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
