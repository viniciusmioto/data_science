# Data Science Case

## Introduction

This is a case study for a Data Science position at 4intelligence. The goal is to analyze the dataset and make predictions for the ABCR Index, which is calculated based on the flow of light vehicles passing through toll plazas in Brazil over recent years (2010 - 2023). Additionally, the dataset includes various explanatory variables that can be used for analysis and forecasting.


* **Understanding Time Series Behavior**: We aim to gain insights into the historical behavior of the ABCR Index over time.
* **Time Series Analysis**: We will perform a comprehensive analysis of the time series data to uncover trends, seasonality, and other patterns.
* **Projection to 2030**: Our ultimate goal is to create accurate forecasts for the ABCR Index up to the end of 2030.

## Repository Content

The **Jupyter Notebook** that contains the code for all the case is in `analysis.ipynb` file. The original **dataset** was saved as `dataset.xslx`.

## Algorithms and Models

For this case, the following algorithms and models were used:
* LSTM
* ARIMA

## Executing the Code

To run the code, you need to have **Python 3** installed. Then follow the steps below:

1. Clone this repository to your local machine.
```bash
git clone https://github.com/viniciusmioto/datascience_case
```

or

```bash 
git clone git@github.com:viniciusmioto/datascience_case.git
```

2. Install the required packages.

Use Python **venv** to create a virtual environment:

```bash
python3 -m venv ./venv
```

Then use pip install:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook.

Open the `analysis.ipynb` and run the code.

I used Jupter Notebook in **VSCode**, so the images might not appear exactly as they should in some IDE's, so I saved all the images in the `images` folder.