import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def preprocess_audio(file, start_time = None, target_fs = 4):
    """
    Pre-process vocalization data from a social partner's audio file into 
    vocalization instances.

    Parameters
    ----------
    file : str
        The filepath of the audio file ending in '.mp3'.
    start_time : datetime.datetime, optional
        A `datetime` value denoting the start time of the audio file. If
        `None`, audio data will be resampled using the mean of values within
        the target sampling interval.
    target_fs : int, optional
        The target sampling rate to which the resample the original signal;
        by default, 4 Hz.

    Returns
    -------
    signal : array_like
        An array containing the pre-processed vocalization signal.
    """
    raw, orig_fs = librosa.load(file, sr = None)
    signal = raw.copy()
    signal[signal != 0] = 1
    if start_time is None:
        rs_factor = orig_fs // target_fs
        reshaped = signal[:len(signal) - len(signal) % rs_factor].reshape(
            -1, rs_factor)
        signal = np.where(np.mean(reshaped, axis = 1) >= 0.5, 1, 0)
    else:
        ts = pd.date_range(start_time, periods = len(signal),
                           freq = pd.Timedelta(seconds = 1 / orig_fs))
        signal = pd.DataFrame({'Signal': signal}).set_index(ts)
        signal = signal.resample(f'{int(1000 / target_fs)}ms').first()
    return signal

def get_vocal_states(p1, p2,  p1_label = 'Child', p2_label = 'Parent', start_time = None, fs = None):
    """
    Process vocalization instances of a dyad into vocal states. Numeric
    values of vocal states are as follows: 1 = Vocalization; 2 = Pause;
    3 = Switching pause; 4 = Non-interruptive simultaneous speech;
    5 = Interruptive simultaneous speech.

    Parameters
    ----------
    p1 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of
        the first partner's vocalizations.
    p2 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of
        the second partner's vocalizations.
    p1_label : str, optional
        The name of the first partner; by default, 'Child'.
    p2_label : str, optional
        The name of the second partner; by default, 'Parent'.
    start_time : datetime.datetime
        A `datetime` value denoting the start time of the vocalization data.
        If `None`, a tuple of two arrays containing the two partners' vocal 
        states will be returned; otherwise, a DataFrame with columns 
        containing timestamps and each partner's vocal states will be 
        returned.
    fs : int
        The sampling rate of the vocalization instances. This value must be 
        provided if `start_time` is not `None`.

    Returns
    -------
    tuple
        If `start_time` is `None`, returns a tuple:
        
        p1_vocal_states : array_like
            An array containing the first partner's processed vocal states.
        p2_vocal_states : array_like
            An array containing the second partner's processed vocal states.

    pandas.DataFrame
        If `start_time` is not `None`, returns a DataFrame with the following 
        columns:
        
        'Timestamp': Timestamped intervals.
        'P1': The first partner's processed vocal states.
        'P2': The second partner's processed vocal states.
    """

    if start_time is not None and fs is None:
        raise ValueError(
            'The sampling rate `fs` must be provided if a start time '
            'is given.')

    # Get all vocal states
    p1_iss, p1_nss, p2_iss, p2_nss = get_simultaneous_speech(p1, p2)
    p1_pauses, p2_pauses = get_pauses(p1, p2)
    p1_switch_pauses, p2_switch_pauses = get_switching_pauses(p1, p2)

    # Pre-process partner 1
    p1_voc = np.where(p1 == 1)[0]
    p1_voc = p1_voc[~np.isin(p1_voc, p1_iss)]
    p1_voc = p1_voc[~np.isin(p1_voc, p1_nss)]
    p1_vocal_states = np.zeros(len(p1), dtype = int)
    if p1_voc.size > 0:
        p1_vocal_states[p1_voc] = 1
    if p1_pauses.size > 0:
        p1_vocal_states[p1_pauses] = 2
    if p1_switch_pauses.size > 0:
        p1_vocal_states[p1_switch_pauses] = 3
    if p1_nss.size > 0:
        p1_vocal_states[p1_nss] = 4
    if p1_iss.size > 0:
        p1_vocal_states[p1_iss] = 5

    # Pre-process partner 2
    p2_voc = np.where(p2 == 1)[0]
    p2_voc = p2_voc[~np.isin(p2_voc, p2_iss)]
    p2_voc = p2_voc[~np.isin(p2_voc, p2_nss)]
    p2_vocal_states = np.zeros(len(p2), dtype = int)
    if p2_voc.size > 0:
        p2_vocal_states[p2_voc] = 1
    if p2_pauses.size > 0:
        p2_vocal_states[p2_pauses] = 2
    if p2_switch_pauses.size > 0:
        p2_vocal_states[p2_switch_pauses] = 3
    if p2_nss.size > 0:
        p2_vocal_states[p2_nss] = 4
    if p2_iss.size > 0:
        p2_vocal_states[p2_iss] = 5

    if start_time is None:
        return p1_vocal_states, p2_vocal_states
    else:
        timestamps = pd.date_range(start = start_time,
                                   freq = f'{1/fs}S',
                                   periods = len(p1_vocal_states))
        dyad = pd.DataFrame({
            'Timestamp': timestamps,
            p1_label: p1_vocal_states,
            p2_label: p2_vocal_states})
        return dyad

def get_vocal_turns(p1, p2, fs = 4, max_pause_duration = 3):
    """
    Identify indices of when each person's vocal turn begins and ends.
    
    Parameters
    ----------
    p1 : array_like
        An array containing the first partner's vocal states.
    p2 : array_like
        An array containing the second partner's vocal states.
    fs : int, float
        The sampling rate of the vocalization instances.
    max_pause_duration : int, float
        The maximum allowable duration of a pause (in seconds) 
        during which a vocal turn is still considered valid.
        
    Returns
    -------
    p1_turns : list
        A list of tuples containing indices denoting the start and end of 
        the first partner's vocal turn.
    p2_turns : list
        A list of tuples containing indices denoting the start and end of
        the second partner's vocal turn.
    """
    if len(p1) != len(p2):
        raise Exception('The lengths of `p1` and `p2` must be equal.')
    
    p1_turns = []
    p2_turns = []
    total_duration = len(p1)
    max_pause = int(max_pause_duration * fs)
    
    n = 0
    while n < total_duration:
        is_turn = False
        
        # Get partner 1's turns
        if p1[n] == 1 and p2[n] == 2:
            p1_start_ix = n
            for m in range(p1_start_ix, total_duration):
                if (p1[m] == 2 or p1[m] == 3) and p2[m] == 2:
                    pause_end = m + max_pause
                    for p in range(m, min(pause_end, total_duration)):
                        if p2[p] == 1 and p1[p] == 2:
                            p2_start_ix = p
                            p1_turns.append((p1_start_ix, p2_start_ix))
                            n = p2_start_ix
                            is_turn = True
                            break
                    if is_turn:
                        break
            if is_turn:
                continue
        
        # Get partner 2's turns
        if p2[n] == 1 and p1[n] == 2:
            p2_start_ix = n
            for m in range(p2_start_ix, total_duration):
                if (p2[m] == 2 or p2[m] == 3) and p1[m] == 2:
                    pause_end = m + max_pause
                    for p in range(m, min(pause_end, total_duration)):
                        if p1[p] == 1 and p2[p] == 2:
                            p1_start_ix = p
                            p2_turns.append((p2_start_ix, p1_start_ix))
                            n = p1_start_ix
                            is_turn = True
                            break
                    if is_turn:
                        break
            if is_turn:
                continue
                
        n += 1
    return p1_turns, p2_turns

def plot_vocals(p1, p2, fs, seg_num = 1, seg_size = 15, 
                p1_label = 'Child', p2_label = 'Parent'):
    """
    Visualize two social partners' vocalization time series.
    
    Parameters
    ----------
    p1 : array_like
        An array containing the first partner's vocalizations.
    p2 : array_like
        An array containing the second partner's vocalizations.
    fs : int
        The sampling rate of the input data.
    seg_num : int, optional
        The segment number to visualize.
    seg_size : int, optional
        The length of the segment (in seconds) to be visualized; by default,
        15 seconds.
    p1_label : str, optional
        The name of the first partner; by default, 'Child'.
    p2_label : str, optional
        The name of the second partner; by default, 'Parent'.
        
    Returns
    -------
    fig : matplotlib.figure
        A figure containing two subplots, one for each social partner's
        vocal states.
    """
    seg_num = seg_num
    seg_start = int((seg_num - 1) * fs * seg_size)
    seg_end = seg_start + int(fs * seg_size)

    seg_start = int((seg_num - 1) * fs * seg_size)
    seg_end = seg_start + int(fs * seg_size)

    p1_voc_on = np.where(np.diff(p1[seg_start:seg_end]) > 0)[0] + 1
    p1_voc_off = np.where(np.diff(p1[seg_start:seg_end]) < 0)[0] + 1
    p2_voc_on = np.where(np.diff(p2[seg_start:seg_end]) > 0)[0] + 1
    p2_voc_off = np.where(np.diff(p2[seg_start:seg_end]) < 0)[0] + 1

    tick_positions = np.arange(0, int(seg_size * fs) + 5, 5)

    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(2, 1, figsize = (10, 3), dpi = 96)
    
    # P1 subplot
    for i in range(len(p1_voc_on) - 1):
        onset = p1_voc_on[i]
        offset = p1_voc_off[i]
        ax[0].add_patch(
            Rectangle(
                (onset, 0), offset - onset, 1, 
                edgecolor = 'salmon', 
                facecolor = 'lightsalmon', 
                alpha = 0.8, 
                lw = 1.5))
    ax[0].set_xlim(0, int(seg_size * fs))
    ax[0].set_xticks([])
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([])
    ax[0].spines['bottom'].set_linewidth(1.7)
    for sp in ['top', 'right', 'left']:
        ax[0].spines[sp].set_visible(False)
    ax[0].set_ylabel(
        p1_label, fontweight = 'bold', fontsize = 13, rotation = 0)
    ax[0].yaxis.set_label_coords(-.04, 0.4)
    
    # P2 subplot
    for i in range(len(p2_voc_on) - 1):
        onset = p2_voc_on[i]
        offset = p2_voc_off[i]
        ax[1].add_patch(
            Rectangle(
                (onset, 0), offset - onset, 1, 
                edgecolor = 'mediumseagreen', 
                facecolor = 'lightseagreen', 
                alpha = 0.8, 
                lw = 1.5))
    ax[1].set_xlim(0, int(seg_size * fs))
    ax[1].set_xticks(tick_positions)
    ax[1].set_xticklabels(
        [f'{int(tick/4) + ((seg_num - 1) * 30)}' for tick in tick_positions], 
        fontsize = 12)
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([])
    ax[1].spines['bottom'].set_linewidth(1.7)
    for sp in ['top', 'right', 'left']:
        ax[1].spines[sp].set_visible(False)
    ax[1].set_ylabel(
        p2_label, fontweight = 'bold', fontsize = 13, rotation = 0)
    ax[1].yaxis.set_label_coords(-.045, 0.4)
    ax[1].set_xlabel('Time [sec]', fontsize = 12, labelpad = 10)
    ax[1].tick_params(
        axis = 'x', which = 'both', width = 1.7, length = 10, pad = 8)

    fig.tight_layout()
    return fig

def get_pauses(p1, p2):
    """
    Identify indices of two social partners' pause occurrences.
    
    Parameters
    ----------
    p1 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the first partner's vocalizations.
    p2 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the second partner's vocalizations.
        
    Returns
    -------
    pauses1 : array_like
        An array containing indices of the first partner's pauses.
    pauses2: array_like
        An array containing indices of the second partner's pauses.
        
    Example
    -------
    >>> p1 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    >>> p2 = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    >>> pauses1, pauses2 = get_pauses(p1, p2)
    """
    
    if len(p1) != len(p2):
        raise ValueError('Input arrays must have the same length.')
    else:
        p1_switches = []
        p2_switches = []
        for n in range(1, len(p1)):
            if p1[n] != p1[n - 1]:
                p1_switches.append(n)
            if p2[n] != p2[n - 1]:
                p2_switches.append(n)
        
        swp1, swp2 = get_switching_pauses(p1, p2)
        swp = [swp1, swp2]
        pauses = {}
        for i, switches in enumerate([p1_switches, p2_switches]):
            pauses[i] = []
            array = p1 if i == 0 else p2
            for n in range(len(switches) - 2):
                sw0 = switches[n]
                sw1 = switches[n + 1]
                sw2 = switches[n + 2]
                if (array[sw0] == 1 and array[sw1] == 0 and array[sw2] == 1):
                    pause_start = sw1
                    pause_end = sw2 - 1
                    pauses[i].append(np.arange(pause_start, pause_end + 1))
            pauses[i] = np.hstack(pauses[i])
            remove = ~np.isin(pauses[i], swp[i])
            pauses[i] = pauses[i][remove]
            
        p1_pauses = pauses[0]
        p2_pauses = pauses[1]
        
        return p1_pauses, p2_pauses
    
def get_switching_pauses(p1, p2):
    """
    Identify indices of two social partners' switching pause occurrences.
    
    Parameters
    ----------
    p1 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the first partner's vocalizations.
    p2 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the second partner's vocalizations.
        
    Returns
    -------
    p1_switching_pauses : array_like
        An array containing indices of the first partner's switching pauses.
    p2_switching_pauses : array_like
        An array containing indices of the second partner's switching pauses.
    """

    if len(p1) != len(p2):
        raise ValueError('Input arrays must have the same length.')
    else:
        p1_switching_pauses = []
        p2_switching_pauses = []

        interaction_len = len(p1)

        for i in range(interaction_len):
            p1_curr = p1[i]
            p2_curr = p2[i]
            p1_prev = p1[i - 1]
            p2_prev = p2[i - 1]

            switch_to_p1 = None
            switch_to_p2 = None

            # if P1 was last to vocalize and both pause
            if not p1_curr and p1_prev and not p2_curr and not p2_prev:
                for j in range(i, interaction_len):
                    p1_curr = p1[j]
                    p2_curr = p2[j]
                    p1_prev = p1[j - 1]
                    p2_prev = p2[j - 1]
                    if not p1_prev and not p2_prev:
                        if p2_curr and not p1_curr:  # p2 gains the turn
                            switch_to_p2 = j
                            break
                        elif p1_curr and not p2_curr:  # p1 vocalizes again
                            break
                    if p1_curr and p2_curr:
                        break
                if switch_to_p2 is not None:
                    p1_switching_pauses.append(np.arange(i, switch_to_p2))

            # if P2 was last to vocalize and both pause
            if not p2_curr and p2_prev and not p1_curr and not p1_prev:
                for j in range(i, interaction_len):
                    p1_curr = p1[j]
                    p2_curr = p2[j]
                    p1_prev = p1[j - 1]
                    p2_prev = p2[j - 1]
                    if not p2_prev and not p1_prev:
                        if p1_curr and not p2_curr:  # p1 gains the turn
                            switch_to_p1 = j
                            break
                        elif p2_curr and not p1_curr:  # p2 vocalizes again
                            break
                    if p1_curr and p1_curr:
                        break
                if switch_to_p1 is not None:
                    p2_switching_pauses.append(np.arange(i, switch_to_p1))

        if len(p1_switching_pauses) != 0:
            p1_switching_pauses = np.hstack(p1_switching_pauses)
        else:
            p1_switching_pauses = np.array(p1_switching_pauses)

        if len(p2_switching_pauses) != 0:
            p2_switching_pauses = np.hstack(p2_switching_pauses)
        else:
            p2_switching_pauses = np.array(p2_switching_pauses)

        return p1_switching_pauses, p2_switching_pauses

def get_simultaneous_speech(p1, p2):
    """
    Identify indices of two social partners' interruptive (ISS) and 
    non-interruptive simultaneous speech (NSS) occurrences.
    
    Parameters
    ----------
    p1 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the first partner's vocalizations.
    p2 : array_like
        An array containing occurrences (`1`) and non-occurrences (`0`) of 
        the second partner's vocalizations.
        
    Returns
    -------
    p1_iss : array_like
        An array containing indices of the first partner's ISS occurrences.
    p1_nss : array_like
        An array containing indices of the first partner's NSS occurrences.
    p2_iss : array_like
        An array containing indices of the second partner's ISS occurrences.
    p2_nss : array_like
        An array containing indices of the second partner's NSS occurrences.
    """

    if len(p1) != len(p2):
        raise ValueError('Input arrays must have the same length.')
    else:
        interaction_len = len(p1)

        p1_start_ix = 0 if p1[0] == 1 else None
        p2_start_ix = 0 if p2[0] == 1 else None
        p1_stop_ix = None
        p2_stop_ix = None

        p1_iss = []
        p1_nss = []
        p2_iss = []
        p2_nss = []

        for i in range(1, interaction_len):
            p1_curr = p1[i]
            p2_curr = p2[i]
            p1_prev = p1[i - 1]
            p2_prev = p2[i - 1]

            # get p1 vocalization boundaries
            if not p1_prev and p1_curr:
                p1_start_ix = i
            if p1_prev and not p1_curr:
                p1_stop_ix = i

            # get p2 vocalization boundaries
            if not p2_prev and p2_curr:
                p2_start_ix = i
            if p2_prev and not p2_curr:
                p2_stop_ix = i

            if p1_start_ix is not None or p2_start_ix is not None:

                try:  # if p2 interjected
                    if p1_start_ix < p2_start_ix and p1_curr and p2_curr:
                        for j in range(i, interaction_len):
                            if p1[j] == 0:
                                p1_stop_ix = j
                                break
                        for j in range(i, interaction_len):
                            if p2[j] == 0:
                                p2_stop_ix = j
                                break

                        # add p2 interruptive simultaneous speech
                        if p1_stop_ix < p2_stop_ix:
                            p2_iss.append(np.arange(p2_start_ix, p1_stop_ix))

                        # add p2 non-interruptive simultaneous speech
                        if p2_stop_ix < p1_stop_ix:
                            p2_nss.append(np.arange(p2_start_ix, p2_stop_ix))

                        p1_stop_ix = None
                        p2_stop_ix = None

                except TypeError:
                    pass

                try:  # if p1 interjected
                    if p2_start_ix < p1_start_ix and p1_curr and p2_curr:
                        for j in range(i, interaction_len):
                            if p1[j] == 0:
                                p1_stop_ix = j
                                break
                        for j in range(i, interaction_len):
                            if p2[j] == 0:
                                p2_stop_ix = j
                                break

                        # add p1 interruptive simultaneous speech
                        if p2_stop_ix < p1_stop_ix:
                            p1_iss.append(np.arange(p1_start_ix, p2_stop_ix))

                        # add p1 non-interruptive simultaneous speech
                        if p1_stop_ix < p2_stop_ix:
                            p1_nss.append(np.arange(p1_start_ix, p1_stop_ix))

                        p1_stop_ix = None
                        p2_stop_ix = None

                except TypeError:
                    pass

            # handle vocalizations at the end
            if i == interaction_len - 1:
                if p2[i] == 1:
                    p2_stop_ix = i + 1
                if p1[i] == 1:
                    p1_stop_ix = i + 1

                # if p2 interjected
                if (p1_start_ix < p2_start_ix) and (
                        p2_start_ix < p2_stop_ix - 1):

                    # add p2 interruptive simultaneous speech
                    if p1_stop_ix < p2_stop_ix:
                        p2_iss.append(np.arange(p2_start_ix, p1_stop_ix))

                    # add p2 non-interruptive simultaneous speech
                    if p2_stop_ix < p1_stop_ix:
                        p2_nss.append(np.arange(p2_start_ix, p2_stop_ix))

                # if p1 interjected
                if (p2_start_ix < p1_start_ix) and (
                        p1_start_ix < p1_stop_ix - 1):

                    # add p1 interruptive simultaneous speech
                    if p2_stop_ix < p1_stop_ix:
                        p1_iss.append(np.arange(p1_start_ix, p2_stop_ix))

                    # add p1 non-interruptive simultaneous speech
                    if p1_stop_ix < p2_stop_ix:
                        p1_nss.append(np.arange(p1_start_ix, p1_stop_ix))

        p1_iss = np.hstack(p1_iss) if len(p1_iss) != 0 else np.array(p1_iss)
        p1_nss = np.hstack(p1_nss) if len(p1_nss) != 0 else np.array(p1_nss)
        p2_iss = np.hstack(p2_iss) if len(p2_iss) != 0 else np.array(p2_iss)
        p2_nss = np.hstack(p2_nss) if len(p2_nss) != 0 else np.array(p2_nss)
        
        return p1_iss, p1_nss, p2_iss, p2_nss