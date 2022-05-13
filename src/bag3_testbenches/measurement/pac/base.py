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


from typing import Any, Optional, Mapping, List, Union, Tuple

from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict

from ..pss.base import PSSTB


class PACTB(PSSTB):
    """This class provide utility methods useful for all PAC simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sweep_options : Mapping[str, Any]
        Dictionary with following entries :
            type : str
                type of sweep (LINEAR / LOG)
            start : Union[str, float]
                initial value of sweep_var
            stop : Union[str, float]
                final value of sweep_var
            num : int
                number of sweep data points
    pac_options : Optional[Mapping[str, Any]]
        Optional.  PAC simulation options dictionary. (spectre -h pac to see available parameters)
    """

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_setup = self.get_pss_sim_setup()
        sweep_options: Mapping[str, Any] = self.specs['sweep_options']
        pac_options: Mapping[str, Any] = self.specs.get('pac_options', {})
        pac_dict = dict(type='PAC',
                       param='freq',
                       sweep=sweep_options,
                       options=pac_options,
                       save_outputs=self.save_outputs,
                       )

        sim_setup['analyses'].append(pac_dict)
        return netlist_info_from_dict(sim_setup)

