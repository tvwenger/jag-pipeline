# jag-pipeline
DRAO JAG Data Reduction Pipeline

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