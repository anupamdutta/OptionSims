import sys
from PyQt5.QtWidgets import QApplication
from app import OptionsSimApp

def test_headless():
    app = QApplication(sys.argv)
    window = OptionsSimApp()
    print("Window created successfully.")
    
    # Run simulation synchronously for test
    from monte_carlo import SimulationConfig, run_monte_carlo
    cfg = SimulationConfig(
        opening_sf=8296.3,
        upside_target=8350.0,
        downside_target=8185.8,
        dte_days=1.621528,
        opening_iv=0.54043,
        r=0.0,
        call_strike=8300.0,
        put_strike=8300.0,
        num_simulations=1000,
        time_steps=200,
        iv_skew_beta=-0.40
    )
    res = run_monte_carlo(cfg)
    print("Simulation finished. Calling UI update handlers...")
    window.on_sim_finished(res)
    print("Summary cards, charts, and table updated successfully!")
    print("Testing table item count:", window.table.rowCount())
    assert window.table.rowCount() > 0
    print("All GUI headless tests PASSED!")

if __name__ == '__main__':
    test_headless()
