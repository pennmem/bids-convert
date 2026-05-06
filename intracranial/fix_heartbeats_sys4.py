import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', 30)
# pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import glob, json

from cmlreaders import get_data_index, CMLReader

from datetime import datetime
from functools import lru_cache
from sklearn.linear_model import LinearRegression


@lru_cache(maxsize=1)
def _r1_data_index():
    """Cached r1 data index. First call loads it from disk; subsequent
    calls in the same process reuse the same DataFrame."""
    return get_data_index('r1', '/')


def ms_to_datetime(mstime):
    # Convert milliseconds to seconds and then to a datetime object
    return datetime.fromtimestamp(mstime / 1000.0)


def _get_field(obj, key):
    """Return obj[key], tolerating obj being a JSON-encoded string or non-dict.

    Some sessions (e.g. R1556J_1 RepFR2) store data['message'] as a stringified
    JSON object rather than a nested dict, which made the previous
    `dict.get(x, key)` call raise "descriptor 'get' for 'dict' objects doesn't
    apply to a 'str' object" on those rows.
    """
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def get_heart(subject, exp, sess, load_host_pc=False, drop_network_test=False, verbose=False):
    # Finds and reads the session log file from the task laptop
    # sess: int, original session number (used in /data10 session_*/ directory)
    log_dir = f'/data10/RAM/subjects/{subject}/behavioral/{exp}/session_{sess}'
    if load_host_pc:
        log_dir += '/elemem/*/event.log'
    else:
        if exp in ['catFR1']:
            log_dir += '/session.json'
        else:
            log_dir += '/session.jsonl'
    log_dir_list = glob.glob(log_dir)
    if verbose:
        print(log_dir_list)
    if len(log_dir_list) != 1: raise ValueError(f'Log not found: {str(log_dir_list)}')
    heart_beat = []
    # read session log to dataframe
    for log_dir in log_dir_list:
        log = []
        with open(log_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                try: log.append(json.loads(line))
                except Exception as e: continue
        temp = pd.DataFrame(log)
        temp['session'] = int(sess)
        heart_beat.append(temp)
    heart_beat = pd.concat(heart_beat)
    # reads the data out of the dict format
    if load_host_pc:
        heart_beat = heart_beat[heart_beat.type.isin(['HEARTBEAT', 'HEARTBEAT_OK'])]
        heart_beat['count'] = heart_beat.data.apply(lambda x: _get_field(x, 'count'))
    else:
        heart_beat['message'] = heart_beat.data.apply(lambda x: _get_field(x, 'message'))
        heart_beat.dropna(subset=['message'], inplace=True)
        heart_beat['type'] = heart_beat.message.apply(lambda x: _get_field(x, 'type'))
        heart_beat['data'] = heart_beat.message.apply(lambda x: _get_field(x, 'data'))
        heart_beat = heart_beat[heart_beat.type.isin(['HEARTBEAT', 'HEARTBEAT_OK'])]
        heart_beat['count'] = heart_beat.data.apply(lambda x: _get_field(x, 'count'))
        if len(heart_beat) == 0: raise ValueError('No HEARTBEAT / HEARTBEAT_OK events logged!')

    # skip heartbeats after initial heartbeat test, which is required to pass before starting a session
    if drop_network_test:
        heart_beat = heart_beat[heart_beat['count'] > 20]

    assert len(heart_beat.session.unique()) == 1, f'session numbers not unique (or not present): {heart_beat.session.unique()}'
    bpm_sent = pd.DataFrame()
    bpm_done = pd.DataFrame()
    bpm_sent = heart_beat[heart_beat.type == 'HEARTBEAT']
    bpm_done = heart_beat[heart_beat.type == 'HEARTBEAT_OK']
    bpm_done.set_index('count', inplace = True)
    bpm_sent.set_index('count', inplace = True)

    # Gets the latency between when the signal is sent and when Elemem sends the signal out
    bpm_err = bpm_done.time.astype(float) - bpm_sent.time.astype(float)
    # bpm_err = bpm_err[bpm_err.index > 20]  # not sure why Leo included this since heartbeats from network test are dropped earlier
    _max = round(bpm_err.max(), 2)
    _min = round(bpm_err.min(), 2)
    ten = round(bpm_err.quantile(.10), 2)
    ninety = round(bpm_err.quantile(.90), 2)
    out_count = bpm_err[bpm_err > 100].count()
    
    hardware_system = 'host_pc' if load_host_pc else 'task_laptop'
    if verbose:
        # print(f'{subject} {exp} session', sess)
        print('HEARTBEAT statistics for', hardware_system)
        print(f'Minimum latency: {_min} ms, Maximum latency: {_max} ms')
        print(f'Tenth percentile latency: {ten} ms. Ninetieth percentile latency: {ninety} ms')
        print(f'Latencies greater than 100 ms: {out_count / len(bpm_err) * 100:0.4}%\n')

    heart_beat = heart_beat.query('type == "HEARTBEAT"')
    if 'message' in heart_beat.columns: 
        heart_beat.drop('message', axis=1, inplace=True)
    heart_beat.set_index('count', inplace=True, drop=False)
    heart_beat.loc[:, ['latency']] = bpm_err
    heart_beat.loc[:, ['time_HEARTBEAT_OK']] = bpm_done.time
    heart_beat.loc[:, ['subject']] = subject
    heart_beat.loc[:, ['experiment']] = exp
    heart_beat.loc[:, ['session']] = sess
    heart_beat.loc[:, ['hardware_system']] = hardware_system
    heart_beat = heart_beat.reindex(columns=['subject', 'experiment', 'session', 'original_session', 'hardware_system',
                                             'count', 'time', 'time_HEARTBEAT_OK', 'latency', 'id'])

    # if verbose:
    #     heart_beat_cp = heart_beat.copy().dropna()
    #     heart_beat_cp['time'] = heart_beat_cp.time.astype(int).apply(ms_to_datetime)
    #     heart_beat_cp['time_HEARTBEAT_OK'] = heart_beat_cp.time_HEARTBEAT_OK.astype(int).apply(ms_to_datetime)
    #     display(heart_beat_cp[['latency', 'time', 'time_HEARTBEAT_OK']].head(25))

    return heart_beat

def correct_event_times(events, offset, slope, time_col='mstime'):
    events = events.copy()
    events[time_col] = events[time_col] * slope + offset
    return events

def get_heartbeat_correction(df, max_latency=2, min_heartbeats=180, max_include_heartbeats=2000, 
                             max_rms_residual=2, ignore_errors=False, verbose=False, plot=False):
    # generate single-session event times corrected for heartbeat latencies
    if len(df.session.unique()) != 1: raise ValueError(f'Heartbeat dataframe contains {len(df.session.unique())} != 1 sessions!')
    
    # Filtering the individual datasets for latency < max_latency ms before merging
    task_laptop_df = df[(df['hardware_system'] == 'task_laptop') & (df['latency'] < max_latency)]
    # host PC "latencies" should be near zero since elemem just immediately sends HEARTBEAT_OK after receiving HEARTBEAT
    host_pc_df = df[(df['hardware_system'] == 'host_pc') & (df['latency'] < 1)]

    merged_df = pd.merge(task_laptop_df, host_pc_df, on='count', suffixes=('_task', '_host'))
    # drop e.g., straggling HEARTBEATs for which no HEARTBEAT_OK was sent back at the end of the task
    merged_df.dropna(subset=['latency_host', 'latency_task'], inplace=True)
    
    n_heartbeats = len(merged_df)
    if n_heartbeats < min_heartbeats:
        raise ValueError(f'Available HEARTBEATs ({len(merged_df)}) < min_heartbeats ({min_heartbeats})')
    elif n_heartbeats > max_include_heartbeats:
        # pick HEARTBEATs at extremes of range
        merged_df = pd.concat([merged_df.iloc[:max_include_heartbeats // 2],
                               merged_df.iloc[-max_include_heartbeats // 2:]])

    # adjust task times to host PC times
    lr = LinearRegression()
    lr.fit(merged_df[['time_task']], merged_df['time_host'])
    slope = lr.coef_[0]
    offset = lr.intercept_
    
    r2 = lr.score(merged_df[['time_task']], merged_df['time_host'])
    residuals = merged_df['time_host'] - lr.predict(merged_df[['time_task']])
    rms_residual = np.sqrt(((residuals)**2).mean())
    
    # adjust offset for network delay (half of the average roundtrip latency)
    average_latency = merged_df['latency_task'].mean()
    latency_correction = average_latency / 2
    adjusted_offset = offset - latency_correction

    if verbose:
        print('Number of low-latency heartbeats:', len(merged_df))
        print(f'R^2: {r2}')
        print(f'RMS residual: {rms_residual:0.6} ms')
        print(f'Slope: {slope:0.6}')
        print(f'Offset: {offset:0.6} ms')
    
    # plot residuals
    if plot:
        plt.hist(-residuals, bins=50)
        plt.title('Timing alignment residuals')
        plt.xlabel('Residual (ms delay from adjusted task laptop times to host PC times)')
    
    if rms_residual > max_rms_residual:
        message = f'RMS residual of linear fit between task and host PC HEARTBEATs ({rms_residual:0.6} ms) > max_rms_residual ({max_rms_residual} ms)'
        if ignore_errors: print(message)
        else: raise ValueError(message)
    if np.abs(slope - 1) > 1e-5:
        message = f'Slope of linear fit between task and host PC HEARTBEATs too far from 1: {slope}'
        if ignore_errors: print(message)
        else: raise ValueError(message)

    # check for host PC events lagging task laptop events after delay adjustment
    # the proportion depends on the assumed network delay correction, but there shouldn't be
    # many events for which the task laptop lags the host PC after correction
    prop_task_lagging_host = (-residuals - latency_correction > 0).mean()
    max_prop_lagging = 0.05
    if prop_task_lagging_host > max_prop_lagging:
        message = f'Proportion of positive residuals after adjustment for network delay: {prop_task_lagging_host}'
        if ignore_errors: print(message)
        else: raise ValueError(message)
    
    res = {
        'uncorrected_offset': offset,
        'offset': adjusted_offset,
        'slope': slope,
        'average_latency': average_latency,
    }
    return res

# confirm that HEARTBEATs are all separated by ~1 second
def check_heartbeat_onsets(heartbeats, min_heartbeat_diff=990, max_heartbeat_diff=1010):
    if len(heartbeats) == 0: return
    sess_cols = ['subject', 'experiment', 'session']
    heartbeats.loc[:, ['time_diff']] = heartbeats[sess_cols + ['time']].groupby(sess_cols).diff().reset_index(drop=True).to_numpy()
    heart_off = np.logical_or(heartbeats.time_diff.dropna() > max_heartbeat_diff, 
                              heartbeats.time_diff.dropna() < min_heartbeat_diff)
    if heart_off.mean():
        print(f'WARNING: some HEARTBEAT events not separated by approximately 1 second '
              f'(acceptable range: [{min_heartbeat_diff}, {max_heartbeat_diff}] ms). '
              f'Proportion of heartbeats outside range: {heart_off.mean()}')



### fix heartbeat entry point

def fix_heartbeats_for_session(subject, experiment, session, events, verbose=False):
    """Apply System-4 heartbeat-derived timing correction to a cml events
    DataFrame (must contain `mstime`).

    Loads task-laptop and host-PC HEARTBEAT logs for the session, fits a
    linear task-time -> host-time mapping via `get_heartbeat_correction`,
    and rewrites `mstime` as `mstime * slope + offset`. Downstream
    `events_to_BIDS()` then derives `onset`, `duration`, etc. from the
    corrected `mstime`.

    Returns the corrected events DataFrame, or the original events
    DataFrame unchanged if the correction can't be computed (logs a
    warning so a single-session failure does not abort the conversion).
    """
    try:
        di = _r1_data_index()
        sel = di.query("subject==@subject & experiment==@experiment & session==@session").iloc[0]
        subject_alias = sel.subject_alias if not pd.isna(sel.subject_alias) else subject
        sess_orig = int(sel.original_session if not pd.isna(sel.original_session) else sel.session)

        hb_task = get_heart(subject_alias, experiment, sess_orig,
                            drop_network_test=True, load_host_pc=False, verbose=verbose)
        hb_task.loc[:, ['session']] = session
        hb_host = get_heart(subject_alias, experiment, sess_orig,
                            drop_network_test=True, load_host_pc=True, verbose=verbose)
        hb_host.loc[:, ['session']] = session
        heartbeats = pd.concat([hb_task, hb_host], ignore_index=True)

        res = get_heartbeat_correction(heartbeats, ignore_errors=True, verbose=verbose, plot=False)
        offset = res['offset']
        slope = res['slope']

        # mstime is an absolute task-laptop timestamp in ms — full correction.
        corrected = correct_event_times(events, offset, slope, time_col='mstime')
        # eegoffset is a sample index whose origin is the EEG file start
        # (already in host-PC reference), so only the slope (sample-rate
        # drift) applies — pass offset=0 to suppress the wall-clock offset.
        # Round back to its original numpy dtype since correct_event_times
        # promotes to float and downstream BIDS treats `sample` as integer.
        eegoffset_dtype = events['eegoffset'].dtype
        corrected = correct_event_times(corrected, 0, slope, time_col='eegoffset')
        corrected['eegoffset'] = corrected['eegoffset'].round().astype(eegoffset_dtype)

        print(f"HEARTBEAT FIX: applied slope={slope:.9f} to eegoffset, "
              f"slope+offset={offset:.3f} ms to mstime "
              f"for {subject}/{experiment}/ses-{session}")
        return corrected
    except Exception as e:
        print(f"WARNING: heartbeat fix failed for {subject}/{experiment}/ses-{session} "
              f"— leaving mstime uncorrected ({type(e).__name__}: {e})")
        return events