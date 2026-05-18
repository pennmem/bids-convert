"""Reusable scoring, aggregation, regression, and visualization helpers
for the non-heartbeat message-alignment analysis.

Shared by check_nonheart_messages.ipynb and
check_nonheart_messages_multi.ipynb.
"""
import json
import glob
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from cmlreaders import get_data_index
from sklearn.linear_model import (
    RANSACRegressor,
    LinearRegression as _LR,
    TheilSenRegressor as _TS,
)


@lru_cache(maxsize=1)
def _r1_data_index():
    return get_data_index('r1', '/')


def _resolve_subject_alias(subject, experiment, session):
    """Look up the subject_alias for (subject, experiment, session) in the
    r1 data index. Falls back to `subject` if the row is missing or the
    alias is NaN."""
    di = _r1_data_index()
    hit = di.query("subject==@subject & experiment==@experiment & session==@session")
    if len(hit) == 0:
        return subject
    alias = hit.iloc[0].subject_alias
    return subject if pd.isna(alias) else alias


# ===== Constants =====

RANSAC_INLIER_MS = 20.0   # |residual| <= this counts as inlier
SESS_SECONDS     = 3600

PEARSON_TOL   = 1e-5
PEARSON_ROUND = int(round(-np.log10(PEARSON_TOL)))

HEARTBEAT_LIKE = {'HEARTBEAT', 'HEARTBEAT_OK'}

AMBIG_PEARSON_THRESHOLD_DEFAULT = 0.999
AMBIG_QUALITY_RATIO_DEFAULT     = 2.0

HEARTBEAT_DIR = '/home1/zrentala/bids-convert/heartbeat/results'
os.makedirs(HEARTBEAT_DIR, exist_ok=True)

CORRECTION_MODES = ('all_events', 'outliers_only')


def _corrected_h(t, h, slope, offset, inlier_mask, mode):
    """Return corrected host timestamps under one of two strategies.

    'all_events':    every event snapped to the fit line (slope*t+offset).
    'outliers_only': only outliers snapped; inliers keep raw h.
    """
    if mode == 'all_events':
        return slope * np.asarray(t, dtype=float) + offset
    if mode == 'outliers_only':
        t = np.asarray(t, dtype=float)
        h = np.asarray(h, dtype=float)
        return np.where(np.asarray(inlier_mask, dtype=bool),
                        h, slope * t + offset)
    raise ValueError(f"Unknown mode {mode!r}; expected one of {CORRECTION_MODES}")


def _safe_name(s):
    """Make `s` safe to drop into a filename."""
    return str(s).replace('/', '_').replace(' ', '_').replace('<', '').replace('>', '')


def _save_fig(fig, name, save_dir=HEARTBEAT_DIR):
    """Save fig to {save_dir}/{name}.png at 300 dpi, tight bbox."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    return path


# ===== Log-reading helpers =====

def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def task_type(rec):
    """Unwrap 'network' records to get the inner message type."""
    t = rec.get('type')
    if t == 'network':
        msg = (rec.get('data') or {}).get('message')
        if isinstance(msg, dict):
            return msg.get('type')
        return None
    return t


def task_time(rec):
    """Prefer inner-message time on network-wrapped records."""
    if rec.get('type') == 'network':
        msg = (rec.get('data') or {}).get('message')
        if isinstance(msg, dict) and 'time' in msg:
            return msg['time']
    return rec.get('time')


def norm(t):
    return None if t is None else str(t).upper()


def find_logs(subject_alias, experiment, original_session):
    base = (f'/data10/RAM/subjects/{subject_alias}/behavioral/{experiment}/'
            f'session_{original_session}')
    task_path = None
    for fn in ('session.jsonl', 'session.json'):
        p = f'{base}/{fn}'
        if glob.glob(p):
            task_path = p
            break
    host_matches = glob.glob(f'{base}/elemem/*/event.log')
    host_path = host_matches[0] if host_matches else None
    return task_path, host_path


def estimate_one_way_latency_ms(host_records):
    hb_send, hb_recv = {}, {}
    for r in host_records:
        t = r.get('type')
        d = r.get('data') or {}
        c = d.get('count') if isinstance(d, dict) else None
        if c is None:
            continue
        if t == 'HEARTBEAT':
            hb_send[c] = r.get('time')
        elif t == 'HEARTBEAT_OK':
            hb_recv[c] = r.get('time')
    pairs = [hb_recv[c] - hb_send[c] for c in hb_send.keys() & hb_recv.keys()
             if hb_send[c] is not None and hb_recv[c] is not None]
    return float(np.median(pairs)) / 2.0 if pairs else 0.0


# ===== Pair scoring =====

def score_pair(t_name, h_name, t_times, h_times, host_dur, latency_ms,
               max_count_ratio=1, min_count=3):
    """Score one (task_type, host_type) pair on a single session.

    Returns a dict of metrics, or None if the pair doesn't meet count
    thresholds.
    """
    nT, nH = len(t_times), len(h_times)
    if min(nT, nH) < min_count:
        return None
    if max(nT, nH) / min(nT, nH) > max_count_ratio:
        return None
    n = min(nT, nH)
    t = np.array(t_times[:n], dtype=float)
    h = np.array(h_times[:n], dtype=float)

    err_before      = h - t
    err_mean_before = float(np.mean(err_before))
    err_std_before  = float(np.std(err_before, ddof=1))
    err_mean_after  = err_mean_before - latency_ms

    pearson = (float(np.corrcoef(t, h)[0, 1])
               if (np.std(t) > 0 and np.std(h) > 0) else float('nan'))

    if len(h_times) >= 3:
        diffs = np.diff(np.array(h_times, dtype=float))
        mean_interval = float(diffs.mean())
        std_interval  = float(diffs.std(ddof=1))
        cv_burst_score = std_interval / mean_interval if mean_interval else float('inf')
    else:
        mean_interval = std_interval = cv_burst_score = float('nan')

    coverage = (((h_times[-1] - h_times[0]) / 1000.0) / host_dur
                if (host_dur and nH >= 2) else float('nan'))

    return {
        'task_type':       t_name,
        'host_type':       h_name,
        'match_type':      'same_name' if t_name == h_name else 'cross_name',
        'n_task':          nT,
        'n_host':          nH,
        'n_pairs':         n,
        'pearson':         pearson,
        'mean_interval':   mean_interval,
        'std_interval':    std_interval,
        'cv_burst_score':  cv_burst_score,
        'coverage':        coverage,
        'err_mean_before': err_mean_before,
        'err_std_before':  err_std_before,
        'err_mean_after':  err_mean_after,
        'err_std_after':   err_std_before,
    }


def score_session(task_path, host_path):
    """Run all-pair scoring for one session.

    Returns (rows, latency_ms, host_dur, task_by_type, host_by_type)."""
    try:
        task = read_jsonl(task_path)
        host = read_jsonl(host_path)
    except Exception:
        return [], 0.0, 0.0, {}, {}
    if not task or not host:
        return [], 0.0, 0.0, {}, {}

    task_pairs = [(norm(task_type(r)), task_time(r)) for r in task]
    host_pairs = [(norm(r.get('type')),  r.get('time'))  for r in host]
    task_pairs = [(t, ts) for t, ts in task_pairs
                  if t and ts is not None and t not in HEARTBEAT_LIKE]
    host_pairs = [(t, ts) for t, ts in host_pairs
                  if t and ts is not None and t not in HEARTBEAT_LIKE]

    task_by_type = {}
    for t, ts in task_pairs:
        task_by_type.setdefault(t, []).append(ts)
    for t in task_by_type:
        task_by_type[t].sort()
    host_by_type = {}
    for t, ts in host_pairs:
        host_by_type.setdefault(t, []).append(ts)
    for t in host_by_type:
        host_by_type[t].sort()

    host_times_all = [ts for _, ts in host_pairs]
    host_dur = ((max(host_times_all) - min(host_times_all)) / 1000
                if host_times_all else 0)
    latency_ms = estimate_one_way_latency_ms(host)

    rows = []
    for tn in sorted(task_by_type):
        for hn in sorted(host_by_type):
            sc = score_pair(tn, hn, task_by_type[tn], host_by_type[hn],
                            host_dur, latency_ms)
            if sc is not None:
                rows.append(sc)
    return rows, latency_ms, host_dur, task_by_type, host_by_type


def score_all_sessions(qualifying, verbose=True):
    """Run score_session over every row of `qualifying`.

    Returns (all_pairs_df, session_event_cache, skipped) where
    session_event_cache is keyed by (subject, experiment, session)."""
    all_pairs = []
    skipped = []
    session_event_cache = {}

    for i, row in qualifying.iterrows():
        sub  = row['subject']
        exp  = row['experiment']
        sess = int(row['session'])
        orig = (int(row['original_session'])
                if not pd.isna(row['original_session']) else sess)
        if verbose:
            print(f'[{i + 1:>3}/{len(qualifying)}]  '
                  f'{sub} / {exp} / ses-{sess} (orig {orig})', end='  ')

        alias = _resolve_subject_alias(sub, exp, sess)
        task_path, host_path = find_logs(alias, exp, orig)
        if task_path is None or host_path is None:
            if verbose:
                print(f'-- SKIP: missing logs (task={task_path}, host={host_path})')
            skipped.append((sub, exp, sess, 'missing logs'))
            continue

        rows, latency_ms, host_dur, task_by_type, host_by_type = score_session(
            task_path, host_path)
        if not rows:
            if verbose:
                print(f'-- SKIP: no scored pairs '
                      f'(latency={latency_ms:.2f} ms, host_dur={host_dur:.1f}s)')
            skipped.append((sub, exp, sess, 'no scored pairs'))
            continue

        n_same  = sum(1 for r in rows if r['match_type'] == 'same_name')
        n_cross = len(rows) - n_same
        if verbose:
            print(f'-- OK: {len(rows)} pairs ({n_same} same-name + {n_cross} cross-name), '
                  f'latency={latency_ms:.2f} ms, host_dur={host_dur:.1f}s')

        session_event_cache[(sub, exp, sess)] = {
            'task_by_type': task_by_type,
            'host_by_type': host_by_type,
            'host_dur':     host_dur,
            'latency_ms':   latency_ms,
        }

        for r in rows:
            r.update({
                'subject':          sub,
                'experiment':       exp,
                'session':          sess,
                'original_session': orig,
                'latency_ms':       latency_ms,
                'host_dur_s':       host_dur,
            })
        all_pairs.extend(rows)

    all_pairs_df = pd.DataFrame(all_pairs)
    if verbose:
        print()
        print(f'sessions scored: {qualifying.shape[0] - len(skipped)} '
              f'/ {qualifying.shape[0]}')
        print(f'total (session, pair) rows: {len(all_pairs_df)}')
        print(f'cached per-session event maps: {len(session_event_cache)}')
        if skipped:
            print(f'\nskipped sessions ({len(skipped)}):')
            for sub, exp, sess, reason in skipped:
                print(f'  {sub} / {exp} / ses-{sess}  ({reason})')

    return all_pairs_df, session_event_cache, skipped


# ===== Aggregation =====

def aggregate_pairs(all_pairs_df, pearson_tol=PEARSON_TOL):
    """Aggregate per (task_type, host_type) across all sessions.

    Pearson values within `pearson_tol` are treated as tied via
    `pearson_bucket`.
    """
    if all_pairs_df.empty:
        return pd.DataFrame()

    n_sessions_total = all_pairs_df[
        ['subject', 'experiment', 'session']
    ].drop_duplicates().shape[0]
    agg = all_pairs_df.groupby(['task_type', 'host_type', 'match_type']).agg(
        n_sessions=('n_pairs', 'count'),
        median_pearson=('pearson', 'median'),
        median_n_pairs=('n_pairs', 'median'),
        median_err_std_before=('err_std_before', 'median'),
        median_err_mean_before=('err_mean_before', 'median'),
        median_mean_interval=('mean_interval', 'median'),
        median_cv_burst=('cv_burst_score', 'median'),
        median_coverage=('coverage', 'median'),
        max_err_std_before=('err_std_before', 'max'),
    ).reset_index()
    agg['session_frac']    = agg['n_sessions'] / n_sessions_total
    agg['pearson_bucket']  = agg['median_pearson'].round(
        int(round(-np.log10(pearson_tol))))

    agg = agg.sort_values(
        by=['n_sessions', 'pearson_bucket', 'median_err_std_before',
            'median_n_pairs', 'median_mean_interval', 'median_coverage'],
        ascending=[False, False, True, False, True, False]
    ).reset_index(drop=True)
    return agg


def aggregate_pairs_per_experiment(all_pairs_df, pearson_tol=PEARSON_TOL):
    """Aggregate per (experiment, task_type, host_type)."""
    if all_pairs_df.empty:
        return pd.DataFrame()

    sess_per_exp = (all_pairs_df[['experiment', 'subject', 'session']]
                    .drop_duplicates()
                    .groupby('experiment')
                    .size()
                    .rename('exp_n_sessions'))

    agg = all_pairs_df.groupby(
        ['experiment', 'task_type', 'host_type', 'match_type']
    ).agg(
        n_sessions=('n_pairs', 'count'),
        median_pearson=('pearson', 'median'),
        median_n_pairs=('n_pairs', 'median'),
        median_err_std_before=('err_std_before', 'median'),
        median_err_mean_before=('err_mean_before', 'median'),
        median_mean_interval=('mean_interval', 'median'),
        median_cv_burst=('cv_burst_score', 'median'),
        median_coverage=('coverage', 'median'),
        max_err_std_before=('err_std_before', 'max'),
    ).reset_index()
    agg = agg.merge(sess_per_exp, on='experiment', how='left')
    agg['session_frac']   = agg['n_sessions'] / agg['exp_n_sessions']
    agg['pearson_bucket'] = agg['median_pearson'].round(
        int(round(-np.log10(pearson_tol))))

    agg = agg.sort_values(
        by=['experiment', 'n_sessions', 'pearson_bucket',
            'median_err_std_before', 'median_n_pairs',
            'median_mean_interval', 'median_coverage'],
        ascending=[True, False, False, True, False, True, False]
    ).reset_index(drop=True)
    return agg


# ===== Ambiguity filter =====

def _keep_clean_winners(df, group_cols, quality_col, ratio):
    """For each group keep only the best row when its runner-up is at
    least `ratio` times worse on `quality_col`. Otherwise drop the
    whole group as too ambiguous."""
    keep_idx = []
    for _, group in df.groupby(group_cols, sort=False):
        srt = group.sort_values(quality_col)
        if len(srt) == 1:
            keep_idx.append(srt.index[0])
        else:
            best   = float(srt[quality_col].iloc[0])
            second = float(srt[quality_col].iloc[1])
            if second >= ratio * max(best, 1e-3):
                keep_idx.append(srt.index[0])
            # else: too ambiguous - drop entire group
    return df.loc[keep_idx]


def filter_unambiguous(agg_per_exp,
                       pearson_threshold=AMBIG_PEARSON_THRESHOLD_DEFAULT,
                       quality_ratio=AMBIG_QUALITY_RATIO_DEFAULT,
                       force_keep=None):
    """Drop ambiguous candidate pairings.

    A pair survives iff it's the clean best for its task_type AND for
    its host_type (both sides agree).

    Same-name pairs (task_type == host_type) are always kept,
    regardless of pearson or ambiguity filtering.

    `force_keep` is an optional iterable of (experiment, task_type,
    host_type) tuples that should also be kept regardless of pearson or
    ambiguity filtering. Forced pairs are added to `unambiguous` with
    `dropped_by` left blank and never appear in `dropped`.

    Returns (unambiguous, dropped) DataFrames.
    """
    if agg_per_exp.empty:
        return pd.DataFrame(), pd.DataFrame()

    force_set = {tuple(p) for p in (force_keep or [])}
    key_cols = ['experiment', 'task_type', 'host_type']
    keys = list(zip(*[agg_per_exp[c] for c in key_cols]))
    is_same_name = agg_per_exp['task_type'] == agg_per_exp['host_type']
    is_forced = pd.Series([k in force_set for k in keys],
                          index=agg_per_exp.index) | is_same_name

    credible = agg_per_exp[
        (agg_per_exp['median_pearson'] >= pearson_threshold) | is_forced
    ].copy()

    task_clean = _keep_clean_winners(credible, ['experiment', 'task_type'],
                                     'median_err_std_before', quality_ratio)
    host_clean = _keep_clean_winners(credible, ['experiment', 'host_type'],
                                     'median_err_std_before', quality_ratio)
    forced_idx = credible.index[is_forced.loc[credible.index]]
    surviving = (task_clean.index
                 .intersection(host_clean.index)
                 .union(forced_idx))
    unambiguous = credible.loc[surviving].reset_index(drop=True)

    dropped = credible.drop(index=surviving).copy()
    dropped['dropped_by'] = ''
    not_task = ~dropped.index.isin(task_clean.index)
    not_host = ~dropped.index.isin(host_clean.index)
    dropped.loc[ not_task &  not_host, 'dropped_by'] = (
        'task_type & host_type both ambiguous')
    dropped.loc[ not_task & ~not_host, 'dropped_by'] = (
        'task_type ambiguous (or not best)')
    dropped.loc[~not_task &  not_host, 'dropped_by'] = (
        'host_type ambiguous (or not best)')

    missing = force_set - set(keys)
    if missing:
        print(f'filter_unambiguous: force_keep entries not found in agg_per_exp '
              f'(ignored): {sorted(missing)}')

    return unambiguous, dropped


def filter_tight(agg, max_err_std=50, min_pearson=0.99999,
                 min_coverage=0.5, min_session_frac=0.5):
    """Stricter filter on the cross-experiment `agg` table."""
    if agg.empty:
        return pd.DataFrame()
    return agg[
        (agg['max_err_std_before'] <= max_err_std) &
        (agg['median_pearson']     >= min_pearson) &
        (agg['median_coverage']    >= min_coverage) &
        (agg['session_frac']       >= min_session_frac)
    ].reset_index(drop=True)


def drill_top_candidate(agg, all_pairs_df):
    """Return (top_row, per_session_df) for the top entry in `agg`."""
    if agg.empty:
        return None, pd.DataFrame()
    top = agg.iloc[0]
    sel = all_pairs_df[
        (all_pairs_df['task_type'] == top['task_type']) &
        (all_pairs_df['host_type'] == top['host_type'])
    ].copy()
    return top, sel


# ===== Regression helpers =====

def _pull_pair(candidate, session_event_cache):
    """Return (t, h) numpy arrays for a candidate, or (None, None) if missing."""
    key = (candidate['subject'], candidate['experiment'], candidate['session'])
    cache = session_event_cache.get(key)
    if cache is None:
        return None, None
    t_times = cache['task_by_type'].get(candidate['task_type'], [])
    h_times = cache['host_by_type'].get(candidate['host_type'], [])
    n = min(len(t_times), len(h_times))
    if n < 3:
        return None, None
    t = np.asarray(t_times[:n], dtype=float)
    h = np.asarray(h_times[:n], dtype=float)
    return t, h


def fit_candidate(candidate, session_event_cache, regressor='ransac',
                  inlier_threshold_ms=RANSAC_INLIER_MS,
                  fits_df=None, label=None):
    """Fit a chosen regressor on one candidate's paired task/host timestamps.

    Returns (fit_dict, fits_df).
      - fit_dict: slope, offset, inlier_mask, t, h, plus ids + summary stats.
                  None if the candidate has <3 paired events or the fit failed.
      - fits_df:  the same DataFrame passed in (or a fresh one) with a new row
                  appended for this fit.
    """
    t, h = _pull_pair(candidate, session_event_cache)
    if t is None:
        return None, (fits_df if fits_df is not None else pd.DataFrame())

    X = t.reshape(-1, 1)
    try:
        if regressor == 'ransac':
            est = RANSACRegressor(estimator=_LR(),
                                  residual_threshold=inlier_threshold_ms,
                                  random_state=0)
            est.fit(X, h)
            slope  = float(est.estimator_.coef_[0])
            offset = float(est.estimator_.intercept_)
            inlier_mask = est.inlier_mask_.copy()
        elif regressor == 'linear':
            est = _LR().fit(X, h)
            slope, offset = float(est.coef_[0]), float(est.intercept_)
            residuals_tmp = h - (slope * t + offset)
            inlier_mask = np.abs(residuals_tmp) <= inlier_threshold_ms
        elif regressor == 'theilsen':
            est = _TS(random_state=0).fit(X, h)
            slope, offset = float(est.coef_[0]), float(est.intercept_)
            residuals_tmp = h - (slope * t + offset)
            inlier_mask = np.abs(residuals_tmp) <= inlier_threshold_ms
        else:
            raise ValueError(f'unknown regressor: {regressor!r}')
    except Exception as e:
        print(f'fit failed for {candidate}: {e}')
        return None, (fits_df if fits_df is not None else pd.DataFrame())

    residuals = h - (slope * t + offset)
    rms_all     = float(np.sqrt((residuals ** 2).mean()))
    rms_inliers = (float(np.sqrt((residuals[inlier_mask] ** 2).mean()))
                   if inlier_mask.any() else float('nan'))

    fit_dict = {
        'label':                label,
        'subject':              candidate['subject'],
        'experiment':           candidate['experiment'],
        'session':              candidate['session'],
        'task_type':            candidate['task_type'],
        'host_type':            candidate['host_type'],
        'regressor':            regressor,
        'slope':                slope,
        'offset':               offset,
        't':                    t,
        'h':                    h,
        'inlier_mask':          inlier_mask,
        'n_pairs':              int(len(t)),
        'n_inliers':            int(inlier_mask.sum()),
        'inlier_frac':          float(inlier_mask.mean()),
        'rms_residual':         rms_all,
        'rms_residual_inliers': rms_inliers,
    }

    summary_cols = ['label', 'subject', 'experiment', 'session',
                    'task_type', 'host_type', 'regressor',
                    'slope', 'offset', 'n_pairs', 'n_inliers',
                    'inlier_frac', 'rms_residual', 'rms_residual_inliers']
    new_row = {k: fit_dict[k] for k in summary_cols}
    if fits_df is None or fits_df.empty:
        fits_df = pd.DataFrame([new_row])
    else:
        fits_df = pd.concat([fits_df, pd.DataFrame([new_row])], ignore_index=True)
    return fit_dict, fits_df


def compute_residuals(t, h, slope, offset,
                      inlier_threshold_ms=RANSAC_INLIER_MS):
    """Low-level: compute residuals = h - (slope * t + offset) and the
    in-band mask (|residual| <= threshold). Pure function; no I/O.

    Use this when you already have t, h, slope, offset in hand (e.g. a
    fit_dict from fit_candidate). Use recalculate_residuals(...) when you
    only have a fit and need to pull a target's events from the cache first.
    """
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)
    residuals = h - (slope * t + offset)
    in_band = np.abs(residuals) <= inlier_threshold_ms
    rms_in = (float(np.sqrt((residuals[in_band] ** 2).mean()))
              if in_band.any() else float('nan'))
    return {
        't':                   t,
        'h':                   h,
        'residuals':           residuals,
        'in_band':             in_band,
        'rms_inliers':         rms_in,
        'inlier_threshold_ms': inlier_threshold_ms,
    }


def recalculate_residuals(fit, target, session_event_cache,
                          inlier_threshold_ms=RANSAC_INLIER_MS):
    """Apply fit's slope/offset to TARGET's paired events from the cache.

    Returns the same dict shape as compute_residuals(), or None if target
    has no paired events.
    """
    t, h = _pull_pair(target, session_event_cache)
    if t is None:
        return None
    return compute_residuals(t, h, fit['slope'], fit['offset'],
                             inlier_threshold_ms=inlier_threshold_ms)


def plot_residuals(recalc, ax=None, title=None):
    """Plot residuals vs time on a single panel. Dashed lines at +/- threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    t         = recalc['t']
    residuals = recalc['residuals']
    in_band   = recalc['in_band']
    thr       = recalc['inlier_threshold_ms']
    t_rel     = (t - t[0]) / 1000.0

    ax.scatter(t_rel[in_band],  residuals[in_band],
               c='C0', s=10, alpha=0.6, label=f'inlier (n={in_band.sum()})')
    ax.scatter(t_rel[~in_band], residuals[~in_band],
               c='C3', s=14, alpha=0.7, label=f'outlier (n={(~in_band).sum()})')
    ax.axhline( thr, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(-thr, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(0,   color='k', linestyle=':', linewidth=0.5)
    ax.set_xlabel('reltime (s)')
    ax.set_ylabel('residual (ms)')
    if title is None:
        title = (f'Residuals    % inliers={in_band.mean():.1%}    '
                 f'RMS(inliers)={recalc["rms_inliers"]:.2f} ms')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_pair_with_fit(fit, recalc, ax=None, title=None):
    """Plot target's paired (task, host) events with fit's regression line
    overlaid. Points colored by in_band (residual under fit)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    t       = recalc['t']
    h       = recalc['h']
    in_band = recalc['in_band']
    slope, offset = fit['slope'], fit['offset']
    t0, h0  = t[0], h[0]
    t_rel   = (t - t0) / 1000.0
    h_rel   = (h - h0) / 1000.0

    ax.scatter(t_rel[in_band],  h_rel[in_band],
               c='C0', s=10, alpha=0.6, label=f'inlier (n={in_band.sum()})')
    ax.scatter(t_rel[~in_band], h_rel[~in_band],
               c='C3', s=14, alpha=0.7, label=f'outlier (n={(~in_band).sum()})')
    t_fit_abs = np.array([t.min(), t.max()])
    h_fit_abs = slope * t_fit_abs + offset
    fit_label = f"fit from {fit.get('label') or fit['task_type']}<->{fit['host_type']}"
    ax.plot((t_fit_abs - t0) / 1000.0, (h_fit_abs - h0) / 1000.0,
            'k-', lw=0.8, label=fit_label)
    ax.set_xlabel('reltime task (s)')
    ax.set_ylabel('reltime host (s)')
    ax.set_title(title or f'Paired (host, task) events over fit line\nslope = {slope:.9f}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig, ax


# ===== Batch evaluation =====

def _sessions_for_candidate(all_pairs_df, exp, t_name, h_name):
    """Return the unique (subject, experiment, session) rows for a candidate."""
    return all_pairs_df[
        (all_pairs_df['experiment'] == exp) &
        (all_pairs_df['task_type']  == t_name) &
        (all_pairs_df['host_type']  == h_name)
    ][['subject', 'experiment', 'session']].drop_duplicates()


def _eval_row(exp, t_name, h_name, fit):
    """Build one corrections_eval row from a fit_dict."""
    return {
        'experiment':           exp,
        'task_type':            t_name,
        'host_type':            h_name,
        'subject':              fit['subject'],
        'session':              fit['session'],
        'n_pairs':              fit['n_pairs'],
        'n_inliers':            fit['n_inliers'],
        'n_outliers':           fit['n_pairs'] - fit['n_inliers'],
        'inlier_frac':          fit['inlier_frac'],
        'slope':                fit['slope'],
        'offset':               fit['offset'],
        'rms_residual':         fit['rms_residual'],
        'rms_residual_inliers': fit['rms_residual_inliers'],
    }


def build_corrections_eval(unambiguous, all_pairs_df, session_event_cache,
                           regressor='ransac',
                           inlier_threshold_ms=RANSAC_INLIER_MS):
    """One fit per (unambiguous candidate, session) at a single threshold."""
    if unambiguous.empty:
        return pd.DataFrame()

    eval_rows = []
    for _, cand in unambiguous.iterrows():
        exp, t_name, h_name = cand['experiment'], cand['task_type'], cand['host_type']
        for _, sess_row in _sessions_for_candidate(all_pairs_df, exp, t_name, h_name).iterrows():
            fit, _ = fit_candidate({
                'subject':    sess_row['subject'],
                'experiment': sess_row['experiment'],
                'session':    sess_row['session'],
                'task_type':  t_name,
                'host_type':  h_name,
            }, session_event_cache, regressor=regressor,
                inlier_threshold_ms=inlier_threshold_ms)
            if fit is None:
                continue
            eval_rows.append(_eval_row(exp, t_name, h_name, fit))
    return pd.DataFrame(eval_rows)


def build_corrections_eval_progressive(unambiguous, all_pairs_df, session_event_cache,
                                       thresholds, min_events=50,
                                       regressor='ransac'):
    """One fit per (unambiguous candidate, session) using a progressive
    inlier-threshold sequence.

    `thresholds` is an ascending list (e.g. [1/8, 1/2, 2, 3, 5]). For each
    session: try the tightest threshold first; if `n_inliers < min_events`,
    retry at the next looser threshold. The first threshold to reach the
    minimum wins; if none do, the loosest successful fit is kept.

    Output has the same columns as `build_corrections_eval` plus
    `inlier_threshold_used_ms` recording which threshold each fit landed on.
    """
    if unambiguous.empty:
        return pd.DataFrame()

    eval_rows = []
    for _, cand in unambiguous.iterrows():
        exp, t_name, h_name = cand['experiment'], cand['task_type'], cand['host_type']
        for _, sess_row in _sessions_for_candidate(all_pairs_df, exp, t_name, h_name).iterrows():
            target = {
                'subject':    sess_row['subject'],
                'experiment': sess_row['experiment'],
                'session':    sess_row['session'],
                'task_type':  t_name,
                'host_type':  h_name,
            }
            fit, used_thr = None, None
            for thr in thresholds:
                f, _ = fit_candidate(target, session_event_cache,
                                     regressor=regressor,
                                     inlier_threshold_ms=thr)
                if f is None:
                    continue
                fit, used_thr = f, thr
                if f['n_inliers'] >= min_events:
                    break
            if fit is None:
                continue
            row = _eval_row(exp, t_name, h_name, fit)
            row['inlier_threshold_used_ms'] = used_thr
            eval_rows.append(row)
    return pd.DataFrame(eval_rows)


def summarize_corrections_eval(corrections_eval):
    """Group corrections_eval by candidate, return per-candidate summary."""
    if corrections_eval.empty:
        return pd.DataFrame()
    summary = corrections_eval.groupby(
        ['experiment', 'task_type', 'host_type']
    ).agg(
        n_sessions_fit=('slope', 'count'),
        median_slope=('slope', 'median'),
        iqr_slope=('slope', lambda s: s.quantile(0.75) - s.quantile(0.25)),
        slope_std=('slope', 'std'),
        median_rms_resid_ms=('rms_residual', 'median'),
        median_rms_inliers_ms=('rms_residual_inliers', 'median'),
        median_inlier_frac=('inlier_frac', 'median'),
        median_n_pairs=('n_pairs', 'median'),
        median_n_inliers=('n_inliers', 'median'),
    ).reset_index()
    summary['median_slope_minus_1'] = summary['median_slope'] - 1
    return summary.sort_values(
        ['experiment', 'n_sessions_fit', 'median_inlier_frac',
         'median_rms_inliers_ms', 'slope_std'],
        ascending=[True, False, False, True, True]
    ).reset_index(drop=True)


# ===== Batch visualization =====

def _representative_session(sel, sort_col='rms_residual_inliers'):
    """Median row by `sort_col`."""
    sel_sorted = sel.sort_values(sort_col).reset_index(drop=True)
    return sel_sorted.iloc[len(sel_sorted) // 2]


def _rep_session_for_experiment(ok_exp, exp_candidates,
                                sort_col='rms_residual_inliers'):
    """Pick a single (subject, session) shared by all candidates in one
    experiment, so per-candidate plots within that experiment are
    comparable.

    Strategy: among sessions present in `ok_exp`, prefer the one that
    has fit data for the MOST of the experiment's candidates; among
    coverage-tied sessions, pick the median by aggregate `sort_col`.

    Returns (subject, session) or None if no candidate has any fit data.
    """
    sub_sess_to_metrics = {}
    for _, c in exp_candidates.iterrows():
        rows = ok_exp[(ok_exp['task_type'] == c['task_type']) &
                      (ok_exp['host_type'] == c['host_type'])]
        for _, r in rows.iterrows():
            key = (r['subject'], r['session'])
            sub_sess_to_metrics.setdefault(key, []).append(r[sort_col])
    if not sub_sess_to_metrics:
        return None
    max_cov = max(len(v) for v in sub_sess_to_metrics.values())
    eligible = {k: v for k, v in sub_sess_to_metrics.items()
                if len(v) == max_cov}
    items = sorted(eligible.items(), key=lambda kv: float(np.median(kv[1])))
    return items[len(items) // 2][0]


def plot_slope_distribution(corrections_eval, unambiguous):
    """Per-experiment slope distribution histograms.

    One figure per experiment, with rows = candidates of that experiment.
    Each subplot histograms the slopes across all sessions where the
    candidate has fit data. Vertical dashed line at slope=1 marks the
    no-drift reference.
    """
    if corrections_eval.empty or unambiguous.empty:
        print('No corrections_eval / unambiguous - nothing to plot.')
        return

    ok = corrections_eval.dropna(subset=['slope'])
    candidates = unambiguous.merge(
        ok[['experiment', 'task_type', 'host_type']].drop_duplicates(),
        on=['experiment', 'task_type', 'host_type']
    )[['experiment', 'task_type', 'host_type']].drop_duplicates() \
        .sort_values(['experiment', 'task_type']).reset_index(drop=True)

    if candidates.empty:
        print('No candidates have fit data.')
        return

    for exp, exp_cands in candidates.groupby('experiment', sort=False):
        n = len(exp_cands)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), squeeze=False)
        for i, (_, c) in enumerate(exp_cands.iterrows()):
            sel = ok[(ok['experiment'] == exp) &
                     (ok['task_type']  == c['task_type']) &
                     (ok['host_type']  == c['host_type'])]
            ax = axes[i, 0]
            ax.hist(sel['slope'], bins=20, alpha=0.7)
            ax.ticklabel_format(useOffset=False, style='plain', axis='x')
            ax.axvline(1.0, color='k', linestyle='--', linewidth=0.7)
            ax.set_xlabel('regression slope')
            ax.set_ylabel('# sessions')
            ax.set_title(
                f'{c["task_type"]} <-> {c["host_type"]}  (n_sess={len(sel)})',
                fontsize=10)
            ax.grid(alpha=0.3)
        fig.suptitle(
            f'RANSAC slope distribution (time_task -> time_host)  |  {exp}',
            fontsize=12, y=1.02)
        plt.tight_layout()
        plt.show()




# ===== Compared-to-GT (heartbeat-based ground-truth) =====


def _lookup_gt(gt_corrections, subject, experiment, session):
    """Return (gt_slope, gt_offset) for a session, or (None, None)."""
    row = gt_corrections[
        (gt_corrections['subject']    == subject) &
        (gt_corrections['experiment'] == experiment) &
        (gt_corrections['session']    == session)
    ]
    if row.empty:
        return None, None
    return float(row.iloc[0]['slope']), float(row.iloc[0]['offset'])


def _filter_to_top_candidates(df, top_candidates):
    """Inner-join `df` on (experiment, task_type, host_type) with
    top_candidates, if provided. Returns df unchanged when top_candidates
    is None / empty."""
    if top_candidates is None or top_candidates.empty:
        return df
    tc = top_candidates[['experiment', 'task_type', 'host_type']].drop_duplicates()
    return df.merge(tc, on=['experiment', 'task_type', 'host_type'], how='inner')


def _slope_eff_outliers_only(t, h, slope_nh, offset_nh,
                             inlier_threshold_ms=RANSAC_INLIER_MS):
    """For outliers_only mode: snap outliers to the non-heart fit line,
    keep inliers as raw h, then refit a slope through the partially-
    corrected (t, corrected_h). Returns (slope_eff, inlier_frac)."""
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)
    if len(t) < 2:
        return float('nan'), float('nan')
    recalc = compute_residuals(t, h, slope_nh, offset_nh,
                               inlier_threshold_ms=inlier_threshold_ms)
    inlier_mask = np.asarray(recalc['in_band'], dtype=bool)
    corrected_h = np.where(inlier_mask, h, slope_nh * t + offset_nh)
    # Least-squares slope through corrected_h vs t.
    slope_eff, _intercept = np.polyfit(t, corrected_h, 1)
    return float(slope_eff), float(inlier_mask.mean())


def compute_drift_per_candidate(corrections_eval, top_candidates=None, *,
                                session_event_cache=None,
                                mode='all_events',
                                session_seconds=SESS_SECONDS,
                                inlier_threshold_ms=RANSAC_INLIER_MS):
    """Scalar-per-(candidate, session) drift in the non-heart message,
    BEFORE vs AFTER the non-heart correction, using `slope_nh` as truth.

    mode='all_events':
        BEFORE_ms = (slope_nh - 1) * session_seconds * 1000
        AFTER_ms  = 0                                            (trivial)

    mode='outliers_only':
        For each session, snap outliers to slope_nh*t+offset_nh, keep
        inliers as raw h, refit a slope `slope_eff` through the partially
        corrected (t, corrected_h):
        BEFORE_ms = (slope_nh  - 1) * session_seconds * 1000
        AFTER_ms  = (slope_eff - 1) * session_seconds * 1000
        Requires session_event_cache.

    Returns a DataFrame with one row per (candidate, subject, session).
    Columns:
        experiment, task_type, host_type, subject, session, slope_nh,
        drift_before_ms, drift_after_ms
        (+ slope_eff, inlier_frac when mode='outliers_only')

    Sample-rate-dependent `drift_*_samples` are intentionally not produced
    here -- the caller should compute them per session using the real
    `eeg.samplerate` (see check_nonheart_messages_gt.ipynb's drift cells).
    """
    if corrections_eval is None or corrections_eval.empty:
        return pd.DataFrame()
    if mode not in CORRECTION_MODES:
        raise ValueError(f"mode must be one of {CORRECTION_MODES}; got {mode!r}")
    if mode == 'outliers_only' and session_event_cache is None:
        raise ValueError("mode='outliers_only' requires session_event_cache")

    ok = corrections_eval.dropna(subset=['slope']).copy()
    ok = _filter_to_top_candidates(ok, top_candidates)
    if ok.empty:
        return pd.DataFrame()

    slope_nh = ok['slope'].astype(float).values
    drift_before_ms = (slope_nh - 1.0) * session_seconds * 1000.0

    # Pull per-session outlier counts straight from corrections_eval
    # (built by build_corrections_eval; always has n_pairs + n_inliers).
    n_pairs    = ok['n_pairs'].astype(int).values    if 'n_pairs'    in ok.columns else np.zeros(len(ok), dtype=int)
    n_inliers  = ok['n_inliers'].astype(int).values  if 'n_inliers'  in ok.columns else n_pairs
    n_outliers = n_pairs - n_inliers
    inlier_frac_ce = (ok['inlier_frac'].astype(float).values
                      if 'inlier_frac' in ok.columns
                      else np.where(n_pairs > 0, n_inliers / np.maximum(n_pairs, 1), 1.0))

    if mode == 'all_events':
        drift_after_ms = np.zeros(len(ok))
        slope_eff      = slope_nh.copy()
    else:  # outliers_only -- refit slope_eff per session
        slope_eff = np.full(len(ok), np.nan)
        for i, (_, r) in enumerate(ok.iterrows()):
            target = {'subject':    r['subject'],
                      'experiment': r['experiment'],
                      'session':    r['session'],
                      'task_type':  r['task_type'],
                      'host_type':  r['host_type']}
            t, h = _pull_pair(target, session_event_cache)
            if t is None or len(t) < 2:
                continue
            se, _ = _slope_eff_outliers_only(
                t, h, float(r['slope']), float(r['offset']),
                inlier_threshold_ms=inlier_threshold_ms)
            slope_eff[i] = se
        drift_after_ms = (slope_eff - 1.0) * session_seconds * 1000.0

    cols = {
        'experiment':      ok['experiment'].values,
        'task_type':       ok['task_type'].values,
        'host_type':       ok['host_type'].values,
        'subject':         ok['subject'].values,
        'session':         ok['session'].astype(int).values,
        'slope_nh':        slope_nh,
        'n_pairs':         n_pairs,
        'n_inliers':       n_inliers,
        'n_outliers':      n_outliers,
        'inlier_frac':     inlier_frac_ce,
        'drift_before_ms': drift_before_ms,
        'drift_after_ms':  drift_after_ms,
    }
    if mode == 'outliers_only':
        cols['slope_eff'] = slope_eff
    return pd.DataFrame(cols).reset_index(drop=True)


def compute_drift_per_candidate_gt(corrections_eval, gt_corrections,
                                   top_candidates=None, *,
                                   session_event_cache=None,
                                   mode='all_events',
                                   session_seconds=SESS_SECONDS,
                                   inlier_threshold_ms=RANSAC_INLIER_MS):
    """Scalar-per-(candidate, session) drift in the non-heart message,
    BEFORE vs AFTER the non-heart correction, using `slope_gt` (heartbeat
    fit) as truth.

    mode='all_events':
        BEFORE_ms = (slope_gt - 1)         * session_seconds * 1000
        AFTER_ms  = (slope_gt - slope_nh)  * session_seconds * 1000

    mode='outliers_only':
        For each session, snap outliers to slope_nh*t+offset_nh, keep
        inliers as raw h, refit a slope `slope_eff` through the partially
        corrected (t, corrected_h):
        BEFORE_ms = (slope_gt  - 1)         * session_seconds * 1000
        AFTER_ms  = (slope_gt  - slope_eff) * session_seconds * 1000
        Requires session_event_cache.

    Returns a DataFrame with one row per (candidate, subject, session)
    that has BOTH a non-heart fit and a GT fit. Columns:
        experiment, task_type, host_type, subject, session,
        slope_nh, slope_gt,
        drift_before_ms, drift_after_ms
        (+ slope_eff, inlier_frac when mode='outliers_only')

    Sample-rate-dependent `drift_*_samples` are intentionally not produced
    here -- the caller should compute them per session using the real
    `eeg.samplerate` (see check_nonheart_messages_gt.ipynb's drift cells).
    """
    if (corrections_eval is None or corrections_eval.empty or
            gt_corrections is None or gt_corrections.empty):
        return pd.DataFrame()
    if mode not in CORRECTION_MODES:
        raise ValueError(f"mode must be one of {CORRECTION_MODES}; got {mode!r}")
    if mode == 'outliers_only' and session_event_cache is None:
        raise ValueError("mode='outliers_only' requires session_event_cache")

    ok = corrections_eval.dropna(subset=['slope']).copy()
    ok = _filter_to_top_candidates(ok, top_candidates)
    if ok.empty:
        return pd.DataFrame()

    gt = (gt_corrections[['subject', 'experiment', 'session', 'slope']]
          .rename(columns={'slope': 'slope_gt'})
          .dropna(subset=['slope_gt']))
    merged = ok.merge(gt, on=['subject', 'experiment', 'session'], how='inner')
    if merged.empty:
        return pd.DataFrame()

    slope_nh = merged['slope'].astype(float).values
    slope_gt = merged['slope_gt'].astype(float).values
    drift_before_ms = (slope_gt - 1.0) * session_seconds * 1000.0

    n_pairs    = merged['n_pairs'].astype(int).values    if 'n_pairs'    in merged.columns else np.zeros(len(merged), dtype=int)
    n_inliers  = merged['n_inliers'].astype(int).values  if 'n_inliers'  in merged.columns else n_pairs
    n_outliers = n_pairs - n_inliers
    inlier_frac_ce = (merged['inlier_frac'].astype(float).values
                      if 'inlier_frac' in merged.columns
                      else np.where(n_pairs > 0, n_inliers / np.maximum(n_pairs, 1), 1.0))

    if mode == 'all_events':
        drift_after_ms = (slope_gt - slope_nh) * session_seconds * 1000.0
        slope_eff      = slope_nh.copy()
    else:  # outliers_only -- refit slope_eff per session
        slope_eff = np.full(len(merged), np.nan)
        for i, (_, r) in enumerate(merged.iterrows()):
            target = {'subject':    r['subject'],
                      'experiment': r['experiment'],
                      'session':    r['session'],
                      'task_type':  r['task_type'],
                      'host_type':  r['host_type']}
            t, h = _pull_pair(target, session_event_cache)
            if t is None or len(t) < 2:
                continue
            se, _ = _slope_eff_outliers_only(
                t, h, float(r['slope']), float(r['offset']),
                inlier_threshold_ms=inlier_threshold_ms)
            slope_eff[i] = se
        drift_after_ms = (slope_gt - slope_eff) * session_seconds * 1000.0

    cols = {
        'experiment':      merged['experiment'].values,
        'task_type':       merged['task_type'].values,
        'host_type':       merged['host_type'].values,
        'subject':         merged['subject'].values,
        'session':         merged['session'].astype(int).values,
        'slope_nh':        slope_nh,
        'slope_gt':        slope_gt,
        'n_pairs':         n_pairs,
        'n_inliers':       n_inliers,
        'n_outliers':      n_outliers,
        'inlier_frac':     inlier_frac_ce,
        'drift_before_ms': drift_before_ms,
        'drift_after_ms':  drift_after_ms,
    }
    if mode == 'outliers_only':
        cols['slope_eff'] = slope_eff
    return pd.DataFrame(cols).reset_index(drop=True)


def pick_rep_per_experiment(records, corrections_eval,
                            sort_col='rms_residual_inliers'):
    """Filter `records` (from compute_per_candidate_vs_gt) down to one
    (subject, session) per
    experiment, picked via `_rep_session_for_experiment(corrections_eval,
    ..., sort_col=sort_col)`. All candidates in that experiment then
    share the same rep session, so per-candidate plots are comparable.
    """
    if not records:
        return []
    by_exp = {}
    for r in records:
        by_exp.setdefault(r['experiment'], []).append(r)

    out = []
    for exp, recs in by_exp.items():
        exp_cands = pd.DataFrame(
            [{'task_type': r['task_type'], 'host_type': r['host_type']}
             for r in recs]).drop_duplicates()
        ok_exp = corrections_eval[corrections_eval['experiment'] == exp]
        rep_key = _rep_session_for_experiment(ok_exp, exp_cands, sort_col=sort_col)
        if rep_key is None:
            continue
        rep_subject, rep_session = rep_key
        out.extend(r for r in recs
                   if r['subject'] == rep_subject
                   and int(r['session']) == int(rep_session))
    return out




def recalculate_residuals_gt(target, session_event_cache, gt_corrections,
                             inlier_threshold_ms=RANSAC_INLIER_MS):
    """Apply the heartbeat-based (GT) slope/offset for target's session to
    target's events. Returns the same dict shape as recalculate_residuals
    (or None if the session has no GT row / no paired events)."""
    gt_slope, gt_offset = _lookup_gt(gt_corrections,
                                     target['subject'],
                                     target['experiment'],
                                     target['session'])
    if gt_slope is None:
        return None
    gt_fit = {
        'slope':     gt_slope,
        'offset':    gt_offset,
        'task_type': target['task_type'],
        'host_type': target['host_type'],
        'label':     'GT',
    }
    return recalculate_residuals(gt_fit, target, session_event_cache,
                                 inlier_threshold_ms=inlier_threshold_ms)


def compute_per_candidate_vs_gt(unambiguous, session_event_cache, gt_corrections,
                                corrections_eval=None, mode='all_events',
                                inlier_threshold_ms=RANSAC_INLIER_MS):
    """For EVERY (candidate, session) that has a GT fit (and, in
    `outliers_only` mode, a non-heart fit in `corrections_eval`), compute
    the residuals of `corrected_h - gt_fit(t)`. `corrected_h` depends on
    `mode`:

      mode='all_events':    corrected_h = h (raw) -- residuals are
                            h - gt_fit(t); matches the original behavior.
      mode='outliers_only': corrected_h snaps outliers under the non-heart
                            fit; inliers stay as raw h. Needs the stored
                            slope/offset from `corrections_eval`.

    Returns a list of dicts:
        experiment, task_type, host_type, subject, session,
        t, h, corrected_h, residuals, in_band, inlier_mask,
        slope (or None), offset (or None), gt_slope, gt_offset,
        rms, rms_inliers, inlier_threshold_ms, mode

    To pick one rep (subject, session) per experiment for plotting, pipe
    through `pick_rep_per_experiment(...)`.
    """
    if unambiguous.empty or gt_corrections.empty:
        return []
    if mode == 'outliers_only' and (corrections_eval is None or corrections_eval.empty):
        raise ValueError("mode='outliers_only' requires a non-empty corrections_eval")

    cands = unambiguous[['experiment', 'task_type', 'host_type']].drop_duplicates()

    if mode == 'all_events':
        # Iterate every (candidate, session) with a GT fit.
        gt = gt_corrections[['subject', 'experiment', 'session']].drop_duplicates()
        rows = cands.merge(gt, on='experiment')
    else:  # outliers_only -- also need a non-heart fit
        ok = corrections_eval.dropna(subset=['slope'])
        rows = ok.merge(cands, on=['experiment', 'task_type', 'host_type'])
        rows = rows.merge(gt_corrections[['subject', 'experiment', 'session']].drop_duplicates(),
                          on=['subject', 'experiment', 'session'])

    out = []
    for _, r in rows.iterrows():
        subject = r['subject']
        exp     = r['experiment']
        session = int(r['session'])
        t_name  = r['task_type']
        h_name  = r['host_type']

        gt_slope, gt_offset = _lookup_gt(gt_corrections, subject, exp, session)
        if gt_slope is None:
            continue

        target = {'subject': subject, 'experiment': exp, 'session': session,
                  'task_type': t_name, 'host_type': h_name}
        t, h = _pull_pair(target, session_event_cache)
        if t is None or len(t) < 3:
            continue
        t = np.asarray(t, dtype=float)
        h = np.asarray(h, dtype=float)

        nh_slope = nh_offset = None
        inlier_mask = np.ones_like(t, dtype=bool)
        if mode == 'all_events':
            corrected_h = h
        else:  # outliers_only
            nh_slope = float(r['slope'])
            nh_offset = float(r['offset'])
            recalc = compute_residuals(t, h, nh_slope, nh_offset,
                                       inlier_threshold_ms=inlier_threshold_ms)
            inlier_mask = recalc['in_band']
            corrected_h = _corrected_h(t, h, nh_slope, nh_offset,
                                       inlier_mask, mode='outliers_only')

        residuals = corrected_h - (gt_slope * t + gt_offset)
        in_band = np.abs(residuals) <= inlier_threshold_ms
        rms = float(np.sqrt((residuals ** 2).mean())) if len(residuals) else float('nan')
        rms_inliers = (float(np.sqrt((residuals[in_band] ** 2).mean()))
                       if in_band.any() else float('nan'))

        out.append({
            'experiment':           exp,
            'task_type':            t_name,
            'host_type':            h_name,
            'subject':              subject,
            'session':              session,
            't':                    t,
            'h':                    h,
            'corrected_h':          corrected_h,
            'residuals':            residuals,
            'in_band':              in_band,
            'inlier_mask':          inlier_mask,
            'slope':                nh_slope,
            'offset':               nh_offset,
            'gt_slope':             gt_slope,
            'gt_offset':            gt_offset,
            'rms':                  rms,
            'rms_inliers':          rms_inliers,
            'inlier_threshold_ms':  inlier_threshold_ms,
            'mode':                 mode,
        })
    return out


def plot_per_candidate_vs_gt(per_cand_data, *,
                             save=True, save_dir=HEARTBEAT_DIR):
    """Per-candidate 2-panel figure (paired events + GT regression line,
    residuals histogram) from precomputed `per_cand_data` (output of
    compute_per_candidate_vs_gt). Plotting only — no math."""
    if not per_cand_data:
        print('No per_cand_data - nothing to plot.')
        return []

    saved_paths = []
    for d in per_cand_data:
        # Build the shapes plot_pair_with_fit / plot_residuals expect.
        gt_fit = {
            'slope':     d['gt_slope'],
            'offset':    d['gt_offset'],
            'task_type': d['task_type'],
            'host_type': d['host_type'],
            'label':     'GT',
        }
        recalc = {
            't':                   d['t'],
            'h':                   d['corrected_h'],
            'residuals':           d['residuals'],
            'in_band':             d['in_band'],
            'rms_inliers':         d['rms_inliers'],
            'inlier_threshold_ms': d['inlier_threshold_ms'],
        }
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        plot_pair_with_fit(gt_fit, recalc, ax=axes[0])
        plot_residuals(recalc, ax=axes[1])
        fig.suptitle(
            f'Residuals vs HEARTBEATs After Fixing {d["mode"]}\n'
            f'{d["experiment"]}  |  {d["task_type"]} <-> {d["host_type"]}  |  '
            f'ses: {d["subject"]}/{d["session"]}  |  ',
            fontsize=12, y=1.02)
        plt.tight_layout()

        if save:
            sub_dir = os.path.join(save_dir, 'per_candidate_vs_gt')
            name = (f'{_safe_name(d["experiment"])}_'
                    f'{_safe_name(d["task_type"])}_'
                    f'{_safe_name(d["host_type"])}_'
                    f'{_safe_name(d["subject"])}_ses{d["session"]}_'
                    f'{d["mode"]}')
            saved_paths.append(_save_fig(fig, name, save_dir=sub_dir))
        plt.show()
    return saved_paths


# ===== Per-candidate session-drift distributions =====

def _plot_drift_distribution(drift_df, top_candidates, unit_label, color,
                             truth_label, mode='all_events', *,
                             session_seconds=SESS_SECONDS, sfreq=None,
                             save=True, save_dir=HEARTBEAT_DIR,
                             plot_type='drift_distribution',
                             quantity_tag='quantity'):
    """One 2-panel histogram per (experiment, candidate) from precomputed
    `drift_df` (output of compute_drift_per_candidate or
    compute_drift_per_candidate_gt). LEFT = drift_before_*, RIGHT =
    drift_after_*. `truth_label` is 'nh' or 'gt' for the suptitle.
    `mode` is 'all_events' or 'outliers_only', for suptitle + save_dir
    disambiguation."""
    if drift_df is None or drift_df.empty:
        print('No drift_df - nothing to plot.')
        return []

    unit_tag    = 'samples' if unit_label == 'samples' else 'ms'
    before_col  = f'drift_before_{unit_tag}'
    after_col   = f'drift_after_{unit_tag}'
    if before_col not in drift_df.columns or after_col not in drift_df.columns:
        raise ValueError(f"drift_df missing columns {before_col}/{after_col}")

    cands = (top_candidates[['experiment', 'task_type', 'host_type']]
             .drop_duplicates()
             if top_candidates is not None and not top_candidates.empty
             else drift_df[['experiment', 'task_type', 'host_type']].drop_duplicates())

    saved_paths = []
    # If sfreq not provided explicitly, fall back to drift_df's per-row
    # 'samplerate' column (added by the notebook's recompute step). Only
    # used for the suptitle annotation in the eegoffset (samples) panel.
    if sfreq is None and 'samplerate' in drift_df.columns:
        _sf = drift_df['samplerate'].dropna()
        sfreq = float(_sf.median()) if len(_sf) else None
    if unit_label == 'samples' and sfreq is not None:
        sfreq_str = f' @ {sfreq:g} Hz'
    else:
        sfreq_str = ''
    for _, c in cands.iterrows():
        sub = drift_df[(drift_df['experiment'] == c['experiment']) &
                       (drift_df['task_type']  == c['task_type'])  &
                       (drift_df['host_type']  == c['host_type'])]
        if sub.empty:
            continue
        before_vals = sub[before_col].to_numpy()
        after_vals  = sub[after_col].to_numpy()
        if unit_label == 's':
            before_vals = before_vals / 1000.0
            after_vals  = after_vals  / 1000.0

        truth_str = f"{c['task_type'][:7]:7} <-> {c['host_type'][:7]:7}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, vals, panel in [(axes[0], before_vals, 'BEFORE'),
                                (axes[1], after_vals,  f'AFTER ({mode})')]:
            ax.hist(vals, bins=30, color=color, alpha=0.7)
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_yscale('log')
            ax.ticklabel_format(useOffset=False, style='plain', axis='x')
            ax.set_xlabel(f'drift ({unit_label})')
            ax.set_ylabel('# sessions')
            med = float(np.median(vals)) if len(vals) else float('nan')
            std = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
            ax.set_title(f'{panel}    median={med:.1f}    std={std:.1f}')
            ax.grid(alpha=0.3)

        if 'n_outliers' in sub.columns:
            med_out = int(round(float(sub['n_outliers'].median())))
            tot_out = int(sub['n_outliers'].sum())
            outlier_str = f'  |  outliers per sess: median={med_out}, total={tot_out}'
        else:
            outlier_str = ''
        fig.suptitle(
            f'Drift over {session_seconds}s{sfreq_str} BEFORE vs AFTER fixing {mode}'
            f'non-heartbeat correction against {truth_label}\n'
            f'{c["experiment"]}  |  {c["task_type"]} <-> {c["host_type"]}  |  '
            f'n_sessions={len(sub)}{outlier_str}',
            fontsize=11, y=1.02)
        plt.tight_layout()

        if save:
            sub_dir = os.path.join(save_dir,
                                   f'{plot_type}_{quantity_tag}_{truth_label}_{mode}')
            name = (f'{_safe_name(c["experiment"])}_'
                    f'{_safe_name(c["task_type"])}_'
                    f'{_safe_name(c["host_type"])}_{truth_label}_{mode}')
            saved_paths.append(_save_fig(fig, name, save_dir=sub_dir))
        plt.show()
    return saved_paths


def plot_eegoffset_drift_distribution(drift_df, top_candidates=None, *,
                                      mode='all_events',
                                      sfreq=None, session_seconds=SESS_SECONDS,
                                      save=True, save_dir=HEARTBEAT_DIR):
    """NH-truth: per-candidate eegoffset drift histogram (samples)."""
    return _plot_drift_distribution(
        drift_df, top_candidates,
        unit_label='samples', color='orange', truth_label='non-HEARTBEAT', mode=mode,
        session_seconds=session_seconds, sfreq=sfreq,
        save=save, save_dir=save_dir, quantity_tag='eegoffset')


def plot_wallclock_drift_distribution(drift_df, top_candidates=None, *,
                                      mode='all_events',
                                      session_seconds=SESS_SECONDS,
                                      save=True, save_dir=HEARTBEAT_DIR):
    """NH-truth: per-candidate wall-clock drift histogram (s)."""
    return _plot_drift_distribution(
        drift_df, top_candidates,
        unit_label='s', color='green', truth_label='non-HEARTBEAT', mode=mode,
        session_seconds=session_seconds,
        save=save, save_dir=save_dir, quantity_tag='wallclock')


def plot_eegoffset_drift_distribution_gt(drift_df, top_candidates=None, *,
                                         mode='all_events',
                                         sfreq=None, session_seconds=SESS_SECONDS,
                                         save=True, save_dir=HEARTBEAT_DIR):
    """GT-truth: per-candidate eegoffset drift histogram (samples).
    BEFORE = drift in raw signal from GT slope; AFTER = drift residual
    after applying the non-heart correction, vs GT."""
    return _plot_drift_distribution(
        drift_df, top_candidates,
        unit_label='samples', color='orange', truth_label='HEARTBEAT', mode=mode,
        session_seconds=session_seconds, sfreq=sfreq,
        save=save, save_dir=save_dir, quantity_tag='eegoffset')


def plot_wallclock_drift_distribution_gt(drift_df, top_candidates=None, *,
                                         mode='all_events',
                                         session_seconds=SESS_SECONDS,
                                         save=True, save_dir=HEARTBEAT_DIR):
    """GT-truth: per-candidate wall-clock drift histogram (s).
    BEFORE = drift in raw signal from GT slope; AFTER = drift residual
    after applying the non-heart correction, vs GT."""
    return _plot_drift_distribution(
        drift_df, top_candidates,
        unit_label='s', color='green', truth_label='HEARTBEAT', mode=mode,
        session_seconds=session_seconds,
        save=save, save_dir=save_dir, quantity_tag='wallclock')


# ===== Mode comparison: all_events vs outliers_only =====

def aggregate_modes_per_session(per_cand_all, per_cand_out, top_candidates):
    """Long DataFrame of per-session summary stats for each (candidate,
    mode), restricted to the candidates in `top_candidates`.

    Each input is a list of dicts from compute_per_candidate_vs_gt(...)
    with `mode` set on every record. Records with empty residual arrays
    are skipped.

    Columns: experiment, task_type, host_type, subject, session, mode,
    rms, mean_resid, std_resid, n_events.
    """
    if top_candidates is None or top_candidates.empty:
        return pd.DataFrame()

    keys = set(zip(top_candidates['experiment'],
                   top_candidates['task_type'],
                   top_candidates['host_type']))

    rows = []
    for records in (per_cand_all, per_cand_out):
        for d in records or []:
            k = (d['experiment'], d['task_type'], d['host_type'])
            if k not in keys:
                continue
            resid = np.asarray(d['residuals'], dtype=float)
            if resid.size == 0:
                continue
            rows.append({
                'experiment': d['experiment'],
                'task_type':  d['task_type'],
                'host_type':  d['host_type'],
                'subject':    d['subject'],
                'session':    int(d['session']),
                'mode':       d['mode'],
                'rms':        float(d['rms']),
                'mean_resid': float(np.mean(resid)),
                'std_resid':  float(np.std(resid)),
                'n_events':   int(resid.size),
            })
    return pd.DataFrame(rows)


def paired_wilcoxon_per_candidate(long_df, stat_col):
    """Per-candidate paired Wilcoxon signed-rank between modes.

    Pivots `long_df` (from aggregate_modes_per_session) on `mode`, keys
    by (subject, session), keeps only sessions present in BOTH modes,
    then runs scipy.stats.wilcoxon on `stat_col`.

    Returns DataFrame: experiment, task_type, host_type, n_paired,
    all_events_mean, outliers_only_mean, statistic, pvalue.
    """
    if long_df is None or long_df.empty or stat_col not in long_df.columns:
        return pd.DataFrame()

    out = []
    for (exp, tt, ht), grp in long_df.groupby(
            ['experiment', 'task_type', 'host_type'], sort=False):
        wide = grp.pivot_table(index=['subject', 'session'],
                               columns='mode', values=stat_col,
                               aggfunc='first')
        if 'all_events' not in wide.columns or 'outliers_only' not in wide.columns:
            continue
        wide = wide.dropna(subset=['all_events', 'outliers_only'])
        n_paired = len(wide)
        if n_paired < 1:
            continue
        blue = wide['all_events'].to_numpy()
        orange = wide['outliers_only'].to_numpy()

        stat = float('nan')
        pval = float('nan')
        if n_paired >= 2 and not np.all(blue == orange):
            try:
                res = stats.wilcoxon(blue, orange, zero_method='wilcox')
                stat = float(res.statistic)
                pval = float(res.pvalue)
            except ValueError:
                pass

        out.append({
            'experiment':           exp,
            'task_type':            tt,
            'host_type':            ht,
            'n_paired':             n_paired,
            'all_events_mean':      float(np.mean(blue)),
            'outliers_only_mean':   float(np.mean(orange)),
            'statistic':            stat,
            'pvalue':               pval,
        })
    return pd.DataFrame(out)


def _sig_stars(p):
    """Significance stars for a p-value."""
    if p is None or not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _fmt_val(v):
    """Compact, scale-adaptive number formatter for bar value labels."""
    if not np.isfinite(v):
        return ''
    a = abs(v)
    if a >= 10000:
        return f'{v:.0f}'
    if a >= 100:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.2f}'
    return f'{v:.3f}'


def _paired_mode_bars(ax, exp_df, cands, stat_col, wilco, exp,
                     labels, title, ylab):
    """Helper: draw paired blue/orange bars per candidate with Wilcoxon
    significance stars from the precomputed `wilco` DataFrame."""
    blue_means = []
    orange_means = []
    stars = []
    for t, h in cands:
        sub = exp_df[(exp_df['task_type'] == t) & (exp_df['host_type'] == h)]
        blue_means.append(float(sub.loc[sub['mode'] == 'all_events', stat_col].mean()))
        orange_means.append(float(sub.loc[sub['mode'] == 'outliers_only', stat_col].mean()))
        w = wilco[(wilco['experiment'] == exp) &
                  (wilco['task_type']  == t) &
                  (wilco['host_type']  == h)]
        stars.append(_sig_stars(w['pvalue'].iloc[0]) if not w.empty else '')

    x = np.arange(len(cands))
    bw = 0.4
    ax.bar(x - bw/2, blue_means,   width=bw, color='C0', label='all_events')
    ax.bar(x + bw/2, orange_means, width=bw, color='C1', label='outliers_only')

    arr = np.array(blue_means + orange_means, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        ymax = max(float(finite.max()), 0.0)
        ymin = min(float(finite.min()), 0.0)
        yrange = (ymax - ymin) if ymax > ymin else max(abs(ymax), 1.0)
        ax.set_ylim(ymin - 0.10 * yrange, ymax + 0.25 * yrange)
        for i, (b, o) in enumerate(zip(blue_means, orange_means)):
            for v, dx, color in ((b, -bw/2, 'C0'), (o, bw/2, 'C1')):
                if not np.isfinite(v):
                    continue
                offset = 0.02 * yrange if v >= 0 else -0.02 * yrange
                ax.text(i + dx, v + offset, _fmt_val(v),
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=7, color=color)
        for i, s in enumerate(stars):
            top = max(blue_means[i], orange_means[i])
            if not np.isfinite(top):
                top = 0.0
            ax.text(i, top + 0.12 * yrange, s, ha='center',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def plot_rms_comparison(long_df, top_candidates, wilcoxon_rms, *,
                        save=True, save_dir=HEARTBEAT_DIR):
    """One figure per experiment, 1x2: pooled per-session RMS histogram
    (blue = all_events, orange = outliers_only) and per-candidate mean
    RMS bar chart with paired Wilcoxon significance stars.

    `wilcoxon_rms` is the precomputed output of
    paired_wilcoxon_per_candidate(long_df, stat_col='rms')."""
    if long_df is None or long_df.empty or top_candidates.empty:
        print('No data - nothing to plot.')
        return []

    saved = []

    for exp, exp_tc in top_candidates.groupby('experiment', sort=False):
        # Candidate list comes from `wilcoxon_rms` so caller-side filtering
        # (e.g. .query("all_events_mean < 500")) propagates to both panels.
        # Order is preserved from `top_candidates`.
        w_exp = wilcoxon_rms[wilcoxon_rms['experiment'] == exp]
        w_keys = set(zip(w_exp['task_type'], w_exp['host_type']))
        cands = [(t, h) for t, h in exp_tc[['task_type', 'host_type']]
                                          .drop_duplicates()
                                          .itertuples(index=False, name=None)
                 if (t, h) in w_keys]
        if not cands:
            continue
        cand_df = pd.DataFrame(cands, columns=['task_type', 'host_type'])
        exp_df = long_df[long_df['experiment'] == exp].merge(
            cand_df, on=['task_type', 'host_type'])
        if exp_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        blue_rms   = exp_df.loc[exp_df['mode'] == 'all_events',    'rms'].to_numpy()
        orange_rms = exp_df.loc[exp_df['mode'] == 'outliers_only', 'rms'].to_numpy()
        ax = axes[0]
        if blue_rms.size:
            ax.hist(blue_rms,   bins=40, color='C0', alpha=0.5,
                    label=f'all_events (n={blue_rms.size})')
        if orange_rms.size:
            ax.hist(orange_rms, bins=40, color='C1', alpha=0.5,
                    label=f'outliers_only (n={orange_rms.size})')
        ax.set_xlabel('RMS (ms)')
        ax.set_ylabel('# (candidate, session)')
        ax.set_title(f'Pooled per-session RMS ({len(cands)} candidates)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

        labels = [f'{t} ->\n{h}' for t, h in cands]
        _paired_mode_bars(
            axes[1], exp_df, cands, 'rms', wilcoxon_rms, exp,
            labels=labels,
            title='Mean RMS per candidate  (Wilcoxon: *<.05 **<.01 ***<.001)',
            ylab='mean RMS across sessions (ms)')

        fig.suptitle(
            f'RMS vs HEARTBEATs When Correcting All Events vs Outliers Only  |  {exp}',
            fontsize=12, y=1.03)
        # plt.yscale('log')
        plt.tight_layout()

        if save:
            sub_dir = os.path.join(save_dir, 'mode_comparison')
            saved.append(_save_fig(fig, f'{_safe_name(exp)}_rms_comparison',
                                   save_dir=sub_dir))
        plt.show()
    return saved


def plot_residual_mean_std_comparison(long_df, top_candidates,
                                      wilcoxon_mean, wilcoxon_std, *,
                                      save=True, save_dir=HEARTBEAT_DIR):
    """One figure per experiment, 1x2: per-candidate mean residual (left)
    and std residual (right). Each panel: paired blue/orange bars with
    paired Wilcoxon significance stars.

    `wilcoxon_mean` and `wilcoxon_std` are precomputed outputs of
    paired_wilcoxon_per_candidate(long_df, stat_col='mean_resid') and
    paired_wilcoxon_per_candidate(long_df, stat_col='std_resid')."""
    if long_df is None or long_df.empty or top_candidates.empty:
        print('No data - nothing to plot.')
        return []

    saved = []

    for exp, exp_tc in top_candidates.groupby('experiment', sort=False):
        # Candidates plotted = intersection of both wilcoxon dfs (so caller
        # filtering on either propagates), ordered by `top_candidates`.
        wm_keys = set(zip(wilcoxon_mean.loc[wilcoxon_mean['experiment'] == exp, 'task_type'],
                          wilcoxon_mean.loc[wilcoxon_mean['experiment'] == exp, 'host_type']))
        ws_keys = set(zip(wilcoxon_std.loc[wilcoxon_std['experiment'] == exp, 'task_type'],
                          wilcoxon_std.loc[wilcoxon_std['experiment'] == exp, 'host_type']))
        keep = wm_keys & ws_keys
        cands = [(t, h) for t, h in exp_tc[['task_type', 'host_type']]
                                          .drop_duplicates()
                                          .itertuples(index=False, name=None)
                 if (t, h) in keep]
        if not cands:
            continue
        cand_df = pd.DataFrame(cands, columns=['task_type', 'host_type'])
        exp_df = long_df[long_df['experiment'] == exp].merge(
            cand_df, on=['task_type', 'host_type'])
        if exp_df.empty:
            continue

        labels = [f'{t} ->\n{h}' for t, h in cands]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        _paired_mode_bars(
            axes[0], exp_df, cands, 'mean_resid', wilcoxon_mean, exp,
            labels=labels,
            title='Mean residual per candidate  (Wilcoxon: *<.05 **<.01 ***<.001)',
            ylab='mean residual (ms)')
        axes[0].axhline(0, color='k', linestyle='--', linewidth=0.7)
        _paired_mode_bars(
            axes[1], exp_df, cands, 'std_resid', wilcoxon_std, exp,
            labels=labels,
            title='Std residual per candidate  (Wilcoxon: *<.05 **<.01 ***<.001)',
            ylab='std residual (ms)')

        fig.suptitle(
            f'Residual Mean & STD vs HEARTBEATs When Correcting All Events vs Outliers Only '
            f'  |  {exp}',
            fontsize=12, y=1.03)
        plt.tight_layout()

        if save:
            sub_dir = os.path.join(save_dir, 'mode_comparison')
            saved.append(_save_fig(fig, f'{_safe_name(exp)}_residual_mean_std',
                                   save_dir=sub_dir))
        plt.show()
    return saved


def plot_per_candidate_fits(corrections_eval, unambiguous, session_event_cache,
                            inlier_threshold_ms=RANSAC_INLIER_MS):
    """For each unambiguous candidate: pick rep session (median
    rms_residual_inliers), refit, show the two-panel (paired-events +
    residuals) figure."""
    if corrections_eval.empty or unambiguous.empty:
        print('No corrections_eval / unambiguous - nothing to plot.')
        return

    ok = corrections_eval.dropna(subset=['slope'])
    candidates = unambiguous[['experiment', 'task_type', 'host_type']].drop_duplicates()

    for exp, exp_cands in candidates.groupby('experiment', sort=False):
        ok_exp = ok[ok['experiment'] == exp]
        rep_key = _rep_session_for_experiment(
            ok_exp, exp_cands, sort_col='rms_residual_inliers')
        if rep_key is None:
            continue
        rep_subject, rep_session = rep_key

        for _, c in exp_cands.iterrows():
            cand = {
                'subject':    rep_subject,
                'experiment': exp,
                'session':    rep_session,
                'task_type':  c['task_type'],
                'host_type':  c['host_type'],
            }
            fit, _ = fit_candidate(cand, session_event_cache, regressor='ransac',
                                   inlier_threshold_ms=inlier_threshold_ms)
            if fit is None:
                continue
            # Self-fit: reuse t, h, slope, offset that fit_candidate already
            # pulled and computed -- no need to re-read the cache.
            recalc = compute_residuals(fit['t'], fit['h'],
                                       fit['slope'], fit['offset'],
                                       inlier_threshold_ms=inlier_threshold_ms)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            plot_pair_with_fit(fit, recalc, ax=axes[0])
            plot_residuals(recalc, ax=axes[1])
            fig.suptitle(
                f'RANSAC fit  |  {exp}  |  '
                f'{c["task_type"]} <-> {c["host_type"]}  |  '
                f'rep ses: {rep_subject}/{rep_session}',
                fontsize=12, y=1.02)
            plt.tight_layout()
            plt.show()


def plot_per_candidate_corrections(corrections_eval, unambiguous,
                                   session_event_cache, sfreq=1000.0):
    """Per-candidate event-correction plots: eegoffset delta (left) and
    mstime drift after correction (right)."""
    if corrections_eval.empty or unambiguous.empty:
        print('No corrections_eval / unambiguous - nothing to plot.')
        return

    ok = corrections_eval.dropna(subset=['slope'])
    candidates = unambiguous[['experiment', 'task_type', 'host_type']].drop_duplicates()

    for exp, exp_cands in candidates.groupby('experiment', sort=False):
        ok_exp = ok[ok['experiment'] == exp]
        rep_key = _rep_session_for_experiment(
            ok_exp, exp_cands, sort_col='rms_residual')
        if rep_key is None:
            continue
        rep_subject, rep_session = rep_key
        cache = session_event_cache.get((rep_subject, exp, rep_session))
        if cache is None:
            continue

        for _, c in exp_cands.iterrows():
            row = ok_exp[(ok_exp['subject']   == rep_subject) &
                         (ok_exp['session']   == rep_session) &
                         (ok_exp['task_type'] == c['task_type']) &
                         (ok_exp['host_type'] == c['host_type'])]
            if row.empty:
                continue
            rep = row.iloc[0]

            t_times = np.array(cache['task_by_type'].get(c['task_type'], []),
                               dtype=float)
            h_times = np.array(cache['host_by_type'].get(c['host_type'], []),
                               dtype=float)
            n = min(len(t_times), len(h_times))
            if n < 3:
                continue
            t_times = t_times[:n]
            h_times = h_times[:n]

            slope  = float(rep['slope'])
            offset = float(rep['offset'])

            t0 = t_times[0]
            eegoffset_orig    = (t_times - t0) * sfreq / 1000.0
            eegoffset_corr    = eegoffset_orig * slope
            delta_eegoffset   = eegoffset_corr - eegoffset_orig

            mstime_orig       = t_times
            mstime_corr       = slope * t_times + offset
            delta_mstime      = mstime_corr - mstime_orig
            rel_delta_mstime  = delta_mstime - delta_mstime[0]

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].plot(eegoffset_orig, delta_eegoffset, '.', alpha=0.6)
            axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
            axes[0].set_xlabel('original eegoffset (samples)')
            axes[0].set_ylabel('Delta eegoffset (samples)')
            axes[0].set_title(
                f'Change in synthetic sample-index per event after slope correction\n'
                f'((slope - 1) * eegoffset_original)\n'
                f'eegoffset correction (slope-only)\nslope = {slope:.9f}')
            axes[0].grid(alpha=0.3)

            axes[1].plot(t_times - t0, rel_delta_mstime, '.', alpha=0.6)
            axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('event time since session start (ms)')
            axes[1].set_ylabel('relative Delta mstime (ms, offset removed)')
            axes[1].set_title(
                f'Slope-induced wall-clock drift over session\n'
                f'(constant cross-system offset removed for visibility)\n'
                f'mstime drift after correction\n'
                f'(constant offset = {delta_mstime[0]:.0f} ms removed)')
            axes[1].grid(alpha=0.3)

            plt.suptitle(
                f'Applied correction: eegoffset Δ (samples @ {sfreq:g} Hz) + '
                f'wall-clock drift (ms)  |  '
                f'{exp}  |  {c["task_type"]} <-> {c["host_type"]}  |  '
                f'rep ses: {rep_subject}/{rep_session}  |  n={n}',
                y=1.02, fontsize=12)
            plt.tight_layout()
            plt.show()

        print(f'  delta eegoffset: min={delta_eegoffset.min():.1f}, '
              f'max={delta_eegoffset.max():.1f}, '
              f'|max|={np.abs(delta_eegoffset).max():.1f} samples')
        print(f'  delta mstime (constant offset): {delta_mstime[0]:.1f} ms')
        print(f'  delta mstime (slope-induced drift): '
              f'min={rel_delta_mstime.min():.3f}, '
              f'max={rel_delta_mstime.max():.3f}, '
              f'|max|={np.abs(rel_delta_mstime).max():.3f} ms')
        print()


def plot_pooled_residual_hist(corrections_eval, unambiguous, session_event_cache,
                              abs_threshold_ms=20,
                              inlier_threshold_ms=RANSAC_INLIER_MS):
    """Per-candidate residual histograms pooled across sessions, using
    recalculate_residuals to apply each session's (slope, offset) back to
    its events."""
    if corrections_eval.empty or unambiguous.empty:
        print('No corrections_eval / unambiguous - nothing to plot.')
        return

    ok = corrections_eval.dropna(subset=['slope'])
    candidates = unambiguous[['experiment', 'task_type', 'host_type']].drop_duplicates()

    for _, c in candidates.iterrows():
        sel = ok[(ok['experiment'] == c['experiment']) &
                 (ok['task_type']  == c['task_type'])  &
                 (ok['host_type']  == c['host_type'])]
        if sel.empty:
            continue

        pooled = []
        for _, sess_row in sel.iterrows():
            target = {
                'subject':    sess_row['subject'],
                'experiment': sess_row['experiment'],
                'session':    sess_row['session'],
                'task_type':  c['task_type'],
                'host_type':  c['host_type'],
            }
            t, h = _pull_pair(target, session_event_cache)
            if t is None:
                continue
            recalc = compute_residuals(t, h,
                                       float(sess_row['slope']),
                                       float(sess_row['offset']),
                                       inlier_threshold_ms=inlier_threshold_ms)
            pooled.extend(recalc['residuals'].tolist())

        pooled = np.array(pooled)
        if pooled.size == 0:
            continue

        n_over   = int((np.abs(pooled) > abs_threshold_ms).sum())
        pct_over = 100 * n_over / pooled.size

        plt.figure(figsize=(8, 5))
        clip = np.abs(pooled) < 1000
        plt.hist(pooled[clip], bins=60, alpha=0.6, label='residual')
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.yscale('log')
        plt.xlabel('residual (ms)')
        plt.ylabel('# events')
        plt.legend()
        plt.title(
            f'Pooled per-event residuals: time_host - (slope*time_task + offset) \n  '
            f'{c["experiment"]}  |  {c["task_type"]} <-> {c["host_type"]}  |  '
            f'{pct_over:.2f}% > {abs_threshold_ms} ms  |  n_evs={pooled.size}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f'  residual median={np.median(pooled):.3f} ms, '
              f'std={pooled.std(ddof=1):.3f} ms, '
              f'p95(|x|)={np.percentile(np.abs(pooled), 95):.3f} ms, '
              f'max(|x|)={np.abs(pooled).max():.3f} ms')
        print()
