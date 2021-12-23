# jag-pipeline
DRAO JAG Data Reduction Pipeline

**This is a work in progress!**

## Installation
```bash
$ pip install --upgrade git+https://github.com/tvwenger/jag-pipeline.git
```
or
```bash
$ git clone https://github.com/tvwenger/jag-pipeline.git
$ cd jag-pipeline
$ pip install --upgrade .
```

## Usage
Pipeline scripts are added to your python environment `bin`. They have the prefix `jagpipe-`.
For example:
```bash
$ jagpipe-combine --help
usage: combine.py [-h] [--version] [-c CHANBIN] [-t TIMEBIN] [-v] outfile datafiles [datafiles ...]

Combine and bin JAG datafiles

positional arguments:
  outfile               Output SDHDF file
  datafiles             Input SDHDF file(s)

optional arguments:
  -h, --help            show this help message and exit
  --version             show program\'s version number and exit
  -c CHANBIN, --chanbin CHANBIN
                        Channel bin size (default: 1)
  -t TIMEBIN, --timebin TIMEBIN
                        Time bin size (default: 1)
  -v, --verbose         Print verbose information (default: False)
```

You can also run the pipeline functions from the python interpreter or other python
programs:

```python
from jagpipe import combine
combine.combine(datafiles, outfile, chanbin=1, timebin=1, verbose=False)
```

## Quick-Start

You probably want to run the pipeline scripts in this order. Be sure to run each with `--help`
first so you can configure them to your needs.

* `jagpipe-combine` to combine multiple SDHDF files into one, and optionally bin in time and/or frequency.
* `jagpipe-flag` to automatically flag interference along the frequency axis.
* `jagpipe-waterfall` to generate a waterfall figure for a SDHDF dataset.
* `jagpipe-setsource` to assign source names to position dataset based on telescope position and to flag off-source integrations.
* `jagpipe-findcal` to identify cal-on integrations and to flag cal-on to cal-off transition integrations.

Also:

* `jagpipe-reset` to reset the cal and flag state of a SDHDF dataset.

## TODO
* Implement HISTORY as appendable dataset in the metadata group
* Deal with scans properly once implemented by JAG filler
* Get rid of `setsource.py` once implemented by JAG filler