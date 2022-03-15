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

* `jagpipe-combine` to combine multiple SDHDF files produced by the JAG filler into one SDHDF file, and optionally bin in time and/or frequency.
* `jagpipe-concat` to concatenate multiple SDHDF files produced by `jagpipe-combine` into one SDHDF file. This is useful if, for example, you had to run `jagpipe-combine`
   on multiple data subsets to conserve disk space.
* `jagpipe-flagchan` to automatically flag interference along the frequency axis.
* `jagpipe-waterfall` to generate a waterfall plot for a SDHDF dataset.
* `jagpipe-findcal` to identify cal-on integrations and to flag cal-on to cal-off transition integrations.
* `jagpipe-flagtime` to automatically flag interference along the time axis. Note that cal-on integrations should already be identified (a la `jagpipe-findcal`)
* `jagpipe-calibrate` to apply known cal-noise temperature and convert measured powers to antenna temperatures.

Also:

* `jagpipe-reset` to reset the cal and flag state of a SDHDF dataset.
* `jagpipe-flagbackup` to backup SDHDF flag tables to an external HD5 file.
* `jagpipe-flagrestore` to restore SDHDF flag tables from an external HD5 file.
* `jagpipe-flagsummary` to print a summary of the flagged data per scan.


## TODO
* Deal with scans properly once implemented by JAG filler
* Get rid of `setsource.py` once implemented by JAG filler

## Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/jag-pipeline).

## License and Warranty

GNU Public License
http://www.gnu.org/licenses/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.