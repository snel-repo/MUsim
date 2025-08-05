import numpy as np
import plotly.graph_objects as go

# load continuous data
# cont_data_path = "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_2.25/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/continuous_20240607-145718_godzilla_20221117_10MU_SNR-1-from_data_jitter-2.25std_method-KS_templates_12-files.dat"
# cont_data_path = "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221117/2022-11-17_17-08-07_myo/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/continuous.dat"  # 0.1949999928474426
# cont_data_path = "/home/smoconn/git/MUsim/continuous_dat_files/continuous_20250310-180512_human_20231003_13MU_SNR-1-from_data_jitter-0.2std_method-KS_templates_1-files.dat"
# cont_data_path = "/snel/share/data/rodent-ephys/open-ephys/human-finger/paper_evals/sim_noise_0.2/Record Node 112/experiment1/recording3/continuous/Acquisition_Board-100.Rhythm Data-B/continuous_20250219-213121_human_20231003_13MU_SNR-1-from_data_jitter-0.2std_method-KS_templates_1-files.dat"
# cont_data_path = "/home/smoconn/git/MUsim/continuous_dat_files/continuous_20250318-174122_human_20231003_10MU_SNR-1-from_data_jitter-0.2std_method-KS_templates_1-files.dat"
# cont_data_path = "/home/smoconn/git/MUsim/continuous_dat_files/continuous_20250624-221428_monkey_20221202_5MU_16CH_SNR-1-from_data_jitter-0.2std_method-KS_templates_1-files.dat"
cont_data_path = "/home/smoconn/git/MUsim/continuous_dat_files/most_recent_continuous.dat"



data = np.fromfile(cont_data_path, "int16")
# data_r = data.reshape((-1, 8))
# data_r = data.reshape((-1, 12))
data_r = data.reshape((-1, 16))
# data_r = data.reshape((-1, 24))

# plot
# fig = px.line(data_r[:120000, :6])
fig = go.Figure()
for ch in range(data_r.shape[1]):
    fig.add_trace(go.Scatter(y=data_r[:300000, ch] - ch * 2000,marker=dict(color='black')))
# fig = px.line(data_r[:120000, [1, 2, 4, 13, 14, 15]])

fig.update_layout(
    yaxis=dict(
        tickmode="array",
        tickvals=np.arange(0, -2000 * data_r.shape[1], -2000),
        ticktext=np.arange(data_r.shape[1]),
    )
)
fig.show()

print(data_r.max(axis=0))
