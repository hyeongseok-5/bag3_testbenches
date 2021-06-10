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

from typing import Any, Optional, Mapping, List, Union, Tuple

from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict

from ..base import GenericTB


class PSSTB(GenericTB):
    """This class provide utility methods useful for all PSS simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_sim : float
            the total simulation time.

    subclasses can define the following optional entries:

    p_port/ n_port : Optional[str]/ Optional[str]
        Optional.  p_port and n_port in autonomous circuit (e.g. free-running VCO)
    fund: Union[float, str]
        Steady state analysis fundamental frequency (or its estimate for autonomous circuits).
    pss_options : Mapping[str, Any]
        Optional.  PSS simulation options dictionary. (spectre -h pss to see available parameters)
    pnoise_probe: Optional[List[Tuple]]
        Optional. A list of pnoise measurment (p_port, n_port, relative harmonic number)
    pnoise_options: Optional[Mapping[str, Any]]
        Optional.  PSS simulation options dictionary. (spectre -h pnoise to see available parameters)
    """

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        # autonomous
        p_port: Optional[str] = specs.get('p_port', None)
        n_port: Optional[str] = specs.get('n_port', None)
        # driven
        fund: Union[float, str] = specs['fund']
        harms: bool = specs['pss_options']['harms']
        pss_default = dict(
            harmonicbalance='yes',
            hbhomotopy='tstab',
            skipdc='yes',
            harms=10,
            saveinit='yes',
            tstab=10e-9,
            errpreset='conservative',
            maxstep=2e-12,
        )
        pnoise_default = dict(
            sweeptype='relative',
            start=1000,
            stop=1e7,
            dec=20,
            maxsideband=harms,
            noiseout='pm',
        )
        pss_options: Mapping[str, Any] = specs.get('pss_options', pss_default)
        pnoise_options: Mapping[str, Any] = specs.get('pnoise_options', pnoise_default)
        pnoise_probe: Optional[List[Tuple]] = specs.get('pnoise_probe', None)
        prio_pss = {1: pss_options, 2: pss_default}
        prio_pnoise = {1: pnoise_options, 2: pnoise_default}

        pss_dict = dict(type='PSS',
                        p_port=p_port,
                        n_port=n_port,
                        fund=fund,
                        options={**prio_pss[2], **prio_pss[1]},
                        save_outputs=self.save_outputs,
                        )
        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [pss_dict]
        if pnoise_probe:
            for probe in pnoise_probe:
                p_port_noise, n_port_noise, relharmnum = probe
                pnoise_dict = dict(type='PNOISE',
                                   freq=fund,
                                   save_outputs=self.save_outputs,
                                   p_port=p_port_noise,
                                   n_port=n_port_noise,
                                   options={'relharmnum': relharmnum, **prio_pnoise[2], **prio_pnoise[1]},
                                   )
                sim_setup['analyses'].append(pnoise_dict)
        return netlist_info_from_dict(sim_setup)

