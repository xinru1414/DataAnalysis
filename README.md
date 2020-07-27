# Amazon Review Data Analysis 

## Installation & Dependencies

[Install packages using pip and virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Then install dependencies uring the requirements.txt.
```
$ pip install -r requirements.txt
```
## Structure
### data
* sample_data.csv
  - A csv file contains amazon review. Must have the following columns: `Text`, `Published`, `Brand`.
  - The `Published` column must contain a string looks like this: `2014-12-16 00:00:00`
* keywords.txt
  - A txt file contains key words for a specific aspect
  - One key word per line.
* colormap.txt
  - A txt file contains a list of colors that will be matched to brands for analysis.
  - One color per line.
  - Number of brands in the csv file should equal to number of colors in this file.
### src
analyze.py
### graphs
contains graphs for each aspect.

## Usage
```
$ python analyze.py
```
command line parameters see analyze.py
