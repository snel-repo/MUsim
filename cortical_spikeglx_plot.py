from pathlib import Path
from pdb import set_trace

import numpy as np
import plotly.graph_objects as go
import spikeglx
from brainbox.io.one import SpikeSortingLoader
from ibllib import io
from one.api import ONE

# from brainbox.io.one import SpikeSortingLoader

# ##
# from one.api import ONE

# ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
# one = ONE(password="international")
# sessions = one.search()
# one.load_cache(tag="2022_Q2_IBL_et_al_RepeatedSite")
# sessions_lab = one.search(lab="mainenlab")
# t0 = 100 # timepoint in recording to stream

##


# cbin_file = Path(
#     "/home/smoconn/data/ZFM-01576/2020-12-01/001/raw_ephys_data/_spikeglx_ephysData_g0_t0.nidq.1e7e8192-99b6-4ded-9d2c-1e383c4e4a28.cbin"
# ).expanduser()
# sr = spikeglx.Reader(cbin_file)
# data_r = sr.read_samples()[0]

# fig = go.Figure()
# for ch in range(data_r.shape[1]):
#     fig.add_trace(go.Scatter(y=data_r[:60000, ch] - ch * 2000))
# # fig = px.line(data_r[:120000, [1, 2, 4, 13, 14, 15]])

# fig.update_layout(
#     yaxis=dict(
#         tickmode="array",
#         tickvals=np.arange(0, -2000 * data_r.shape[1], -2000),
#         ticktext=np.arange(data_r.shape[1]),
#     )
# )
# fig.show()


one = ONE(base_url="https://openalyx.internationalbrainlab.org")
t0 = 100  # timepoint in recording to stream

pid = "da8dfec1-d265-44e8-84ce-6ae9c109b8bd"
ssl = SpikeSortingLoader(pid=pid, one=one)
# The channels information is contained in a dict table / dataframe
channels = ssl.load_channels()

# Get AP and LFP spikeglx.Reader objects
sr_lf = ssl.raw_electrophysiology(band="lf", stream=True)
sr_ap = ssl.raw_electrophysiology(band="ap", stream=True)
