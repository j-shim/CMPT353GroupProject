# The Data Hunters

This is data analysis on predicting adult wage (and partial works on other data) by [Sean Nam](#authors) and [June Shim](#authors) as a group project for *CMPT 353: Computational Data Science (Summer 2019)* from [Simon Fraser University](https://www.sfu.ca/), Burnaby, British Columbia, Canada.

In this project, we applied machine learning techniques with several python libraries such as pandas and sklearn in order to answer/predict given questions.

Detailed instructions for data analysis on [Usage](#usage) section below.

## Setup

You will need Git, Python 3.5+, Jupyter Notebook installed on your machine.

### Cross-platform Install with [Anaconda](https://www.anaconda.com/distribution/) (Windows, macOS, Linux) - Recommended

* Select Python 3.5+ Installer
* Download Git [here](https://git-scm.com/)

### Debian/Ubuntu based Linux with APT

Open up a Terminal and type:

```bash
sudo apt update
sudo apt install python3 python3-dev python3-pip git
pip3 install scipy matplotlib bokeh pandas statsmodels scikit-learn scikit-image numexpr jupyter
```

### macOS with [Homebrew](https://brew.sh/) Package Manager

Open up a Terminal and type:

```bash
brew update
brew install python3 git
pip3 install scipy matplotlib bokeh pandas statsmodels scikit-learn scikit-image numexpr jupyter
```

If you need to install additional packages, install with pip3:

```bash
pip3 install <package-to-install>
```

### Cloning this repository onto your local machine

Open up a Terminal, `cd` to your preferred directory and type:

```bash
git clone git@csil-git1.cs.surrey.sfu.ca:jys2/the-data-hunters.git
```

*Note:* If `git clone` fails, confirm that your [SSH Key](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2) is set up and registered properly.

## Usage

### adult-wage/

This is our main topic/focus of data analysis. The question we want to answer here is to predict whether an adult's yearly income is greater than $50,000 USD, based on many features/information about the person.

Open by typing in Terminal:

```bash
jupyter-notebook adult-wage-prediction.ipynb
```

and run the cells step by step.

### movie-wikidata/

This is our first attempt on data analysis; however, we decided to change our datasets/questions since the prediction scores were not the best, and it was difficult to extract useful insights from the data.

The question we want to answer here is to predict review scores of movies based on various features such as casts, directors and plots.

##### predict_review_by_plots_*.py

Step 1: Extract/Clean data and save to gzipped json

```bash
# Running the program without arguments will display usage:
$ python3 predict_review_by_plots_step1_join_tables.py
Usage: python3 program.py <input_directory> <output_json_gz>
  e.g. python3 program.py data plots.json.gz
```

```bash
# Running below will produce plots.json.gz, which is a result of
# joining Pandas Dataframe and dropping unnecessary columns.
# Note: First argument must be 'data' which is an input folder, and
# second argument must be .json.gz extension
python3 predict_review_by_plots_step1_join_tables.py data plots.json.gz
```

Step 2a: Analyze data by rounding review scores

* Running the program without arguments will display similar message as above:

```bash
# Assuming the output of Step 1 is plots.json.gz,
python3 predict_review_by_plots_step2a_rounding.py plots.json.gz
```

Step 2b: Analyze data with regression

* Run the program similar to Step 2a:

```bash
# Assuming the output of Step 1 is plots.json.gz,
python3 predict_review_by_plots_step2b_regression.py plots.json.gz
```

##### RT_casts_and_directors.ipynb

Open by typing in Terminal:

```bash
jupyter-notebook RT_casts_and_directors.ipynb
```

and run the cells step by step.

### unused-data-analysis/ (credit card fraud detection)

The question we want to answer here is to predict whether a credit card transaction is fraud or not, based on information about the transaction.

* Note that this data is not suitable for our project, as the data (features) is already processed with PCA. Data is archived into this folder.

Open by typing in Terminal:

```bash
jupyter-notebook creditcard.ipynb
```

and run the cells step by step.

## Authors

* **Sean Nam** - shnam@sfu.ca / [GitHub](https://github.com/seannam1218)
* **June Shim** - jys2@sfu.ca / [GitHub](https://github.com/j-shim)

## Acknowledgments

* README template adapted from https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
* movie-wikidata data source - [Gregory Baker](https://www.cs.sfu.ca/~ggbaker/) (Instructor for the course)
* adult-wage data source - http://archive.ics.uci.edu/ml/datasets/Adult
* unused-data-analysis (creditcard.csv) - https://www.kaggle.com/mlg-ulb/creditcardfraud
