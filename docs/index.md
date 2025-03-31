<span style="font-size: 16pt; font-weight: bold">pyvocals 0.1.1</span>  

**pyvocals** is a Python tool for analyzing vocal turn-taking in 
conversational speech. It extracts structured features from audio files 
based on the behavioral coding schema in [[1]](#reference-1), including 
vocalization, pause, simultaneous speech, switching turn, and interruptive 
turn events.

### Features
- Extracts vocal turn-taking features from speech audio files  
- Supports common formats like WAV and MP3  
- Visualizes vocalization time series of a dyad

### Installation
**Using HTTPS:**
<pre>
pip install git+https://github.com/nmy2103/pyvocals.git
</pre>

**Using SSH:**
<pre style="margin-bottom: 2em">
pip install git+ssh://git@github.com/nmy2103/pyvocals.git
</pre>

### References
<a id="reference-1"></a>
[1] Jaffe, J., & Feldstein, S. (1970). _Rhythms of dialogue_. Academic Press.