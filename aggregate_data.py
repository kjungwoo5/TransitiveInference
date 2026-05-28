from pathlib import Path

import pandas as pd
import os

# Load trial data by session using information in session topology
def load_trial_data_by_session(session_path: Path, home_dir: Path, td_df_query=None) -> pd.DataFrame:
    session_topology = pd.read_csv(session_path)
    session_topology.dropna(how='any', inplace=True)

    td_path_pattern = '<name>/TrialData'

    abs_td_paths = session_topology['tdata_file'].to_list()

    sessnames = [Path(td_info.replace('_TrialData', '')).stem
                 for td_info in abs_td_paths]

    for abs_td_path in abs_td_paths:
        if not os.path.exists(abs_td_path):
            print(f'Warning: td file not found: {abs_td_path}')
    
    td_dfs = {sessname:pd.read_csv(abs_td_path) for sessname, abs_td_path in zip(sessnames,abs_td_paths) if os.path.exists(abs_td_path)}
        
    td_df = pd.concat(list(td_dfs.values()), keys=td_dfs.keys(), names=['session_name'], axis=0)
    td_df.reset_index(level=('session_name'), inplace=True)
    if td_df_query:
        td_df = td_df.query(td_df_query)
    return td_df



