from pathlib import Path

import pandas as pd
import os

# IO Utils copied from XdetectionCore:
class LazyPupilLoader:
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.available_keys = self._get_keys()

    def _get_keys(self):
        if not self.store_path.exists():
            return []
        # Derives keys from folder names: session_id=NAME
        return [d.name.split('=')[1] for d in self.store_path.glob('session_id=*')]

    def __getitem__(self, key):
        """Returns an object with a .pupildf attribute"""
        if key not in self.available_keys:
            raise KeyError(f"Session {key} not found.")
        
        # 1. Load the actual DataFrame
        df = pd.read_parquet(
            self.store_path, 
            filters=[('session_id', '==', key)],
            engine='pyarrow'
        )

    def keys(self):
        return self.available_keys

def load_pupil_sess_lazy(store_path: Path):
    """
    Replaces the original pickle load.
    Returns a lazy-loading proxy instead of a full dictionary.
    """
    store_path = Path(store_path).with_suffix('')  # Ensure we point to the folder

    if store_path.is_dir():
        print(f"Initializing lazy loader for Parquet store at {store_path}")
        return LazyPupilLoader(store_path)
    else:
        print(f"Store {store_path} not found. Returning empty dict.")
        return {}


# Load trial data by session using information in session topology
def load_aggregate_trial_data(session_path: Path, home_dir: Path, td_df_query = None) -> pd.DataFrame:
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


# Load pupil data from parquet files by session using session topology
def load_aggregate_pupil_df(session_path: Path, stage: int, parquet_dir: Path, pupil_df_query = None) -> pd.DataFrame:
    """Load pupil data for all sessions in session_topology matching the requested stage."""
    session_topology = pd.read_csv(session_path)
    session_topology.dropna(how='any', inplace=True)
    print(f'Loading pupil data for Stage {stage} from parquet directory...')

    session_topology = session_topology.dropna(how='all').reset_index(drop=True)

    relevant_sessions = session_topology[session_topology['Stage'] == stage].copy()
    if relevant_sessions.empty:
        raise ValueError(f'No sessions found for STAGE={stage}')

    relevant_sessions = relevant_sessions.dropna(subset=['sound_bin']).reset_index(drop=True)
    session_keys = [Path(str(sound_bin).replace('_SoundData', '')).stem
                    for sound_bin in relevant_sessions['sound_bin']]

    pupil_store = load_pupil_sess_lazy(parquet_dir)
    if not hasattr(pupil_store, 'keys') or len(pupil_store.keys()) == 0:
        raise FileNotFoundError(f'Parquet store not found or empty at {parquet_dir}')

    pupil_dfs = []
    missing_sessions = []
    for sess_key in session_keys:
        try:
            pupil_dfs.append(pupil_store[sess_key].pupildf)
        except KeyError:
            missing_sessions.append(sess_key)

    if missing_sessions:
        print(f'Warning: the following sessions were not found in the parquet store: {missing_sessions}')

    if not pupil_dfs:
        raise ValueError('No pupil DataFrames could be loaded for the requested stage.')

    pupil_df = pd.concat(pupil_dfs, axis=0)
    if pupil_df_query:
        pupil_df = pupil_df.query(pupil_df_query)
    
    return pupil_df


# Load harp write data by session using session topology
def load_aggregate_harp_df(session_path: Path, stage: int, harp_dir: Path, harp_df_query = None) -> pd.DataFrame:
    """Load all sound_index files for sessions in the session_topology into one DataFrame."""
    session_topology = pd.read_csv(session_path)
    session_topology.dropna(how='any', inplace=True)
    print(f'Loading harp write data for Stage {stage} from harp directory...')
    
    session_topology = session_topology.dropna(how='all').reset_index(drop=True)
    relevant_sessions = session_topology[session_topology['Stage'] == stage].copy()
    if relevant_sessions.empty:
        raise ValueError(f'No sessions found for STAGE={stage}')
    
    relevant_sessions = relevant_sessions.dropna(subset=['sound_bin']).reset_index(drop=True)

    sound_bin_col = next((col for col in session_topology.columns if col.lower() == 'sound_bin'), None)
    if sound_bin_col is None:
        raise ValueError('session_topology must contain a sound_bin column')

    session_topology = session_topology.dropna(subset=[sound_bin_col]).reset_index(drop=True)
    if session_topology.empty:
        raise ValueError('No valid sound_bin values found in session_topology.')

    aggregated = []
    missing_files = []
    for _, row in relevant_sessions.iterrows():
        session_id = Path(row[sound_bin_col]).stem
        sound_index_file = harp_dir / f"{session_id}_write_indices.csv"
        if not sound_index_file.is_file():
            missing_files.append(str(sound_index_file))
            continue

        df = pd.read_csv(sound_index_file)
        df['session_id'] = session_id.replace('_SoundData', '')
        aggregated.append(df)

    if missing_files:
        print(f'Warning: missing sound_index files for {len(missing_files)} sessions:')
        for missing in missing_files:
            print(f'  {missing}')

    if not aggregated:
        raise ValueError('No sound_index files were loaded from the harp directory.')

    harp_df = pd.concat(aggregated, axis=0, ignore_index=True)
    if harp_df_query:
        harp_df = harp_df.query(harp_df_query)
    
    return harp_df
