from pathlib import Path
import TransitiveInference.behaviour.transinf_behaviour as tb

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitiveinference_pilot.csv")
HOME_DIR = Path(r"C:\bonsai\data\JungWoo")
STAGE = 3

if __name__ == "__main__":
    td_df = tb.load_trial_data_by_session(SESSION_PATH, HOME_DIR)
    tb.plot_reaction_time(td_df)
    tb.plot_X_A_time(td_df, STAGE)