from pathlib import Path
import behaviour as tfb
import data_io as tfio


SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_cleaned.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
STAGE = 5

if __name__ == "__main__":
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    # tfb.plot_reaction_time(td_df = td_df, output_path = OUTPUT_PATH)
    # tfb.plot_X_A_time(td_df = td_df, stage = STAGE, output_path = OUTPUT_PATH)
    
    tfb.plot_learning(td_df, OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])