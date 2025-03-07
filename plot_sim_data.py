import numpy as np
import plotly.express as px

# load continuous data
cont_data_path = "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_2.25/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/continuous_20240607-145718_godzilla_20221117_10MU_SNR-1-from_data_jitter-2.25std_method-KS_templates_12-files.dat"
# cont_data_path = "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221117/2022-11-17_17-08-07_myo/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/continuous.dat"  # 0.1949999928474426

data = np.fromfile(cont_data_path, "int16")
data_r = data.reshape((-1, 8))
# data_r = data.reshape((-1, 24))

# plot
fig = px.line(data_r[:120000, :6])
# fig = px.line(data_r[:120000, [1, 2, 4, 13, 14, 15]])
fig.show()

print(data_r.max(axis=0))
