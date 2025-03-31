# pyvocals 0.1.1
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14933999.svg)](https://doi.org/10.5281/zenodo.14933999)

**pyvocals** is a Python tool for analyzing vocal turn-taking in 
conversational speech. It extracts structured features from audio files 
based on the behavioral coding schema in [[1]](#reference-1), including 
vocalization, pause, simultaneous speech, switching turn, and interruptive 
turn events.

## Features
- Extracts vocal turn-taking features from speech audio files
- Supports common formats like WAV and MP3
- Visualizes vocalization time series of a dyad

## Installation
### Using HTTPS
You can install `pyvocals` directly from this repo using `pip`:

```sh
pip install git+https://github.com/nmy2103/pyvocals.git
```

### Using SSH
```sh
pip install git+ssh://git@github.com/nmy2103/pyvocals.git
```

### For Developers
For developers who want to **modify the package and test changes**, use `-e` 
(editable mode):

```sh
git clone https://github.com/nmy2103/pyvocals.git
cd pyvocals
pip install -e .
```

## Citation
If you use this package in your research, please cite it with the following:
```
@software{pyvocals2025,
  author       = {Yamane, N.},
  title        = {pyvocals: A Python package for vocal turn-taking feature extraction},
  year         = {2025},
  version      = {0.1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14933999},
  url          = {https://doi.org/10.5281/zenodo.14933999}
}
```

## References
<a id="reference-1"></a>
[1] Jaffe, J., & Feldstein, S. (1970). _Rhythms of dialogue_. Academic Press.
