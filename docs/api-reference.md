<span style="font-size: 18pt; font-weight: bold">API Reference</span>

This section provides detailed documentation for the functions available in
this project. To get started, import the functions using
<code>import pyvocals</code>.

<br>

# extract_features

<div class="source-link">[<a href="https://github.com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L366" 
target="_blank">source</a>]</div>

<span class="function-text">extract_features(p1, p2, p1_label = 'Child', 
p2_label = 'Parent', start_time = None, fs = None)</span>

Extract switching and interruptive turn events for each social partner.

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the first partner's vocalizations.
</div>
**p2** (_array-like_)
<div id="api-param-desc">An 
array containing occurrences (<code>1</code>) and non-occurrences 
(<code>0</code>) of the second partner's vocalizations.</div>
**p1_label** (_str, optional_)
<div id="api-param-desc">The name of the first partner; by default, 
<code>'Child'</code>.</div>
**p2_label** (str, optional_)
<div id="api-param-desc">The name of the second partner; by default, 
<code>'Parent'</code>.</div>
**start_time** (_datetime.datetime, optional_)
<div id="api-param-desc">A <code>datetime</code> value denoting the 
start time of the vocalization data.</div>
**fs** (_int, optional_)
<div id="api-param-desc" class="last">The sampling rate of the vocalization 
instances. This value must be provided if <code>start_time</code> is not 
<code>None</code>.</div>

### Returns
**dyad_vocals** (_pandas.DataFrame_)
<div id="api-param-desc" class="last">A DataFrame containing each 
partner's time series of extracted vocal states and turn-taking features.</div>

### Example
<pre>
> p1 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
> p2 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
> start_time = datetime.datetime(2025, 2, 23, 13, 23, 0)
> vocal_turns = pyvocals.extract_features(p1, p2, start_time = start_time, 
                                          fs = 1)
> vocal_turns.head()

<font color="#8c9f87"><b>Output:</b></font>
            Timestamp  Child  Parent  Child_ST  Child_IT  Parent_ST  Parent_IT
0 2025-02-23 12:31:43      0       1       NaN       NaN        NaN        NaN
1 2025-02-23 12:31:44      0       1       NaN       NaN        NaN        NaN
2 2025-02-23 12:31:45      0       1       NaN       NaN        NaN        NaN
3 2025-02-23 12:31:46      5       1       NaN       1.0        NaN        NaN
4 2025-02-23 12:31:47      1       0       NaN       1.0        NaN        NaN
</pre>

<hr>

# preprocess_audio()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L10" target="_blank">source</a>]</div>

<span class="function-text">preprocess_audio(file, start_time = None, 
target_fs = 4)</span>

Pre-process vocalization data from a social partner's audio file into 
vocalization instances.

### Parameters
**file** (_str_)
<div id="api-param-desc">The filepath of the audio file (.wav, .mp3).</div>
**start_time** (_datetime.datetime, optional_)
<div id="api-param-desc">A <code>datetime</code> value denoting the start 
time of the audio file. If <code>None</code>, audio data will be resampled 
using the mean of values within the target sampling interval.</div>
**target_fs** (_int, optional_)
<div id="api-param-desc" class="last">The target sampling rate to which the 
resample the original signal; by default, 4 Hz.</div>

### Returns
**signal** (_array-like_)
<div id="api-param-desc" class="last">An array containing the 
pre-processed vocalization signal.</div>

<hr>

# get_vocal_states()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L57" target="_blank">source</a>]</div>

<span class="function-text">get_vocal_states(p1, p2, p1_label = 'Child', 
p2_label = 'Parent', start_time = None, fs = None)</span>

Process vocalization instances of a dyad into vocal states. Numeric values 
of vocal states are as follows:

- 1 = Vocalization
- 2 = Pause
- 3 = Switching pause
- 4 = Non-interruptive simultaneous speech
- 5 = Interruptive simultaneous speech

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the first partner's vocalizations.</div>
**p2** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the second partner's vocalizations.</div>
**p1_label** (_str, optional_)
<div id="api-param-desc">The name of the first partner; by default, 
<code>'Child'</code>.
</div>
**p2_label** (_str, optional_)
<div id="api-param-desc">The name of the second partner; by default, 
<code>'Parent'</code>.</div>
**start_time** (_datetime.datetime, optional_)
<div id="api-param-desc">A <code>datetime</code> value denoting the start 
time of the vocalization data.</div>
**fs** (_int, optional_)
<div id="api-param-desc" class="last">The sampling rate of the vocalization 
instances. This value must be provided if <code>start_time</code> is not 
<code>None</code>.</div>

### Returns
<font color="#8c9f87">**tuple:**</font> If <code>start_time = None</code>, 
returns:

**p1_vocal_states** (_array-like_)
<div id="api-param-desc">An array containing the first partner's processed 
vocal states.</div>
**p2_vocal_states** (_array-like_)
<div id="api-param-desc" class="last">An array containing the second 
partner's processed vocal states.</div>

<font color="#8c9f87">**pandas.DataFrame:**</font> If 
<code>start_time</code> is not <code>None</code>, returns a DataFrame with 
the following columns:

- 'Timestamp': Timestamped intervals.
- 'P1': The first partner's processed vocal states.
- 'P2': The second partner's processed vocal states.

### Example
<pre style="margin-bottom: 2em;">
> p1 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
> p2 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
> start_time = datetime.datetime(2025, 2, 23, 13, 23, 0)
> vocal_states = pyvocals.get_vocal_states(p1, p2, start_time = start_time, 
                                           fs = 1)
> vocal_states.head()

<font color="#8c9f87"><b>Output:</b></font>
            Timestamp  Child  Parent
0 2025-02-23 12:31:43      0       1
1 2025-02-23 12:31:44      0       1
2 2025-02-23 12:31:45      0       1
3 2025-02-23 12:31:46      5       1
4 2025-02-23 12:31:47      1       0
</pre>

### References
Jaffe, J., & Feldstein, S. (1970). Rhythms of dialogue. Academic Press.

<hr>

# find_vocal_turns()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L175" 
target="_blank">source</a>]</div>

<span class="function-text">find_vocal_turns(p1, p2, fs = 4, 
max_pause_duration = 5)</span>

Identify indices of when each person's switching and interruptive turns 
begin and end.

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing the first partner's vocal 
states.</div>
**p2** (_array-like_)
<div id="api-param-desc">An array containing the second partner's vocal 
states.</div>
**fs** (_int, float_)
<div id="api-param-desc">The sampling rate of the vocalization instances; 
by default, 4 Hz.</div>
**max_pause_duration** (_int, float_, optional)
<div id="api-param-desc" class="last">The maximum allowable duration of a 
pause (in seconds) during which a vocal turn is still considered valid; by 
default, 5.</div>

### Returns
<font color="#8c9f87">**tuple:**</font> 

**p1_switching_turns** (_list_)
<div id="api-param-desc">A list of tuples containing indices denoting the 
start and end of the first partner's switching turns.</div>
**p1_interrupt_turns** (_list_)
<div id="api-param-desc">A list of tuples containing indices denoting the 
start and end of the first partner's interruptive turns.</div>
**p2_switching_turns** (_list_)
<div id="api-param-desc">A list of tuples containing indices denoting the 
start and end of the second partner's switching turns.</div>
**p2_interrupt_turns** (_list_)
<div id="api-param-desc" class="last">A list of tuples containing indices 
denoting the start and end of the second partner's interruptive turns.</div>

<hr>

# find_pauses()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L543" target="_blank">source</a>]</div>

<span class="function-text">find_pauses(p1, p2)</span>

Identify indices of two social partners' pause occurrences.
    
### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the first partner's vocalizations.</div>
**p2** (_array-like_)
<div id="api-param-desc" class="last">An array containing occurrences 
(<code>1</code>) and non-occurrences (<code>0</code>) of the second 
partner's vocalizations.</div>

### Returns
**pauses1** (_array-like_)
<div id="api-param-desc">An array containing indices of the first partner's 
pauses.</div>
**pauses2** (_array-like_)
<div id="api-param-desc" class="last">An array containing indices of the 
second partner's pauses.</div>

<hr>

# find_switching_pauses()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L610" target="_blank">source</a>]</div>

<span class="function-text">find_switching_pauses(p1, p2)</span>

Identify indices of two social partners' switching pause occurrences. A 
switching pause is defined as a pause bounded by the end of one partner's 
vocalization and the start of the other partner's vocalization.

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the first partner's vocalizations.</div>
**p2** (_array-like_)
<div id="api-param-desc" class="last">An array containing occurrences 
(<code>1</code>) and non-occurrences (<code>0</code>) of the second 
partner's vocalizations.</div>

### Returns
<font color="#8c9f87">**tuple:**</font> 

**p1_switching_pauses** (_array-like_)
<div id="api-param-desc">An array containing indices of the first partner's 
switching pauses.</div>
**p2_switching_pauses** (_array-like_)
<div id="api-param-desc" class="last">An array containing indices of the 
second partner's switching pauses.</div>

<hr>

# find_simultaneous_speech()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L699" target="_blank">source</a>]</div>

<span class="function-text">find_simultaneous_speech(p1, p2)</span>

Identify indices of two social partners' interruptive (ISS) and 
non-interruptive simultaneous speech (NSS) occurrences.

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing occurrences (<code>1</code>) and 
non-occurrences (<code>0</code>) of the first partner's vocalizations.</div>
**p2** (_array-like_)
<div id="api-param-desc" class="last">An array containing occurrences 
(<code>1</code>) and non-occurrences (<code>0</code>) of the second 
partner's vocalizations.</div>

### Returns
<font color="#8c9f87">**tuple:**</font> 

**p1_iss** (_array-like_)
<div id="api-param-desc">An array containing indices of the first partner's 
ISS occurrences.</div>
**p1_nss** (_array-like_)
<div id="api-param-desc">An array containing indices of the first partner's 
NSS occurrences.</div>
**p2_iss** (_array-like_)
<div id="api-param-desc">An array containing indices of the 
second partner's ISS occurrences.</div>
**p2_nss** (_array-like_)
<div id="api-param-desc" class="last">An array containing indices of the 
second partner's NSS occurrences.</div>

<hr>

# plot_vocals()

<div class="source-link">[<a href="https://github.
com/nmy2103/pyvocals/blob/main/pyvocals/pyvocals.py#L440" target="_blank">source</a>]</div>

<span class="function-text">plot_vocals(p1, p2, fs, seg_num = 1, seg_size = 
15, p1_label = 'Child', p2_label = 'Parent')</span>

Visualize two social partners' vocalization time series.

### Parameters
**p1** (_array-like_)
<div id="api-param-desc">An array containing the first partner's 
vocalizations.</div>
**p2** (_array-like_)
<div id="api-param-desc">An array containing the second partner's 
vocalizations.</div>
**fs** (_int_)
<div id="api-param-desc">The sampling rate of the input data.</div>
**seg_num** (_int, optional_)
<div id="api-param-desc">The segment number to visualize.</div>
**seg_size** (_int, optional_)
<div id="api-param-desc">The length of the segment (in seconds) to be 
visualized; by default, 15 seconds.</div>
**p1_label** (_str, optional_)
<div id="api-param-desc">The name of the first partner; by default, 'Child'.
</div>
**p2_label** (_str, optional_)
<div id="api-param-desc" class="last">The name of the second partner; by 
default, 'Parent'.</div>

### Returns
**fig** (_matplotlib.figure_)
<div id="api-param-desc" class="last">A figure containing two subplots, one 
for each social partner's vocal states.</div>