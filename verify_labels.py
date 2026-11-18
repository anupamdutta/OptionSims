import sys
from PyQt5.QtWidgets import QApplication
from app import OptionsSimApp
from monte_carlo import SimulationConfig, run_monte_carlo

app = QApplication(sys.argv)
window = OptionsSimApp()

# Check MetricCard font sizes
print('card title font-size:', window.card_up_call_mean.title_label.styleSheet())
print('card value font-size:', window.card_up_call_mean.value_label.styleSheet())
print('card sub font-size:', window.card_up_call_mean.sub_label.styleSheet())

cfg = SimulationConfig(
    opening_sf=8296.3, upside_target=8350.0, downside_target=8185.8,
    dte_days=1.621528, opening_iv=0.54043, r=0.0, call_strike=8300.0, put_strike=8300.0,
    num_simulations=1000, time_steps=200, iv_skew_beta=-0.08
)
res = run_monte_carlo(cfg)
window.on_sim_finished(res)

axes = window.fig_dist.get_axes()
print('Number of distribution subplots:', len(axes))
for i, ax in enumerate(axes):
    print(f"Subplot {i}: xlabel = '{ax.get_xlabel()}' | ylabel = '{ax.get_ylabel()}'")
    assert 'Price' in ax.get_xlabel() or 'Option' in ax.get_xlabel()
    assert 'Count' in ax.get_ylabel() or 'Frequency' in ax.get_ylabel() or 'Paths' in ax.get_ylabel()

print('ALL ENHANCEMENTS VERIFIED!')
