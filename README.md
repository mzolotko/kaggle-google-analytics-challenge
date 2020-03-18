Solution to the Google Analytics Customer Revenue Prediction challenge
==============================


This project is a solution to the Google Analytics Customer Revenue Prediction challenge that helped me secure the top 2% ranking on the private leaderboard. Please refer to the official competition web page (https://www.kaggle.com/c/ga-customer-revenue-prediction) to learn about the task and available data.

This competition was peculiar in many ways. There was a significant change of how the task was formulated half-way through the competition due a revealed data leak (though the "leaked" data available via the Google Analytics demo account were not available directly, they allowed some participants to achieve outstanding scores on the public LB). Not only was the task changed as the result of the leak, new data was made available as well. Besides, there were a couple of other issues that were addressed in the forum and considered valid by the organisers, but for them it was too late to change the rules. These issues are:

- Choice of the main target variable (totals.transactionRevenue, which includes only the last transaction in a session, rather than totals.totalTransactionRevenue, which includes all transactions in a session). Given that transactions are generally very rare, this may have had an insignificant effect on the final results.
- Calculation of the evaluation metric. RMSE was supposed to be calculated using the log figures of the revenue. To my mind taking logs of a monetary figure makes little sense, i.e. large deviations from true monetary values should be penalised proportionally to the small deviations, and not disproportionally mildly.

The final results (at least in the top 50) were relatively close to each other, and it also seems that even the best-performing models don't produce particularly good predictions, though it is difficult to see directly because RMSE is not a relative measure. Anyway it seems that the distribution of the final ranks in the, say top-50 is to a certain extent attributed to luck, that's also why it is probably better to be skeptical about "secret" or "crucial" tricks to achieve this or that ranking.

My approach was relatively straightforward. I tried to make sense of all the data that seemed to contribute value, I conducted a lot of visual/exploratory analysis (not provided in this repo). After feature selection and some feature recoding/generation, the data was aggregated on several time windows so that the features (X) relate to a certain period of time, and the target variable y relates (relative to this period of time) to a certain future period of time. The aggregation could have been possible in many ways, and it could have been possible to treat various aggregation parameters as hyperparameters that can be optimised, but due to lack of time I stuck to one parameter combination. The ultimate model I used was GBM (LightGBM) without hyperparameter optimisation (also due to lack of time). Bottom line is that this project was more of a data wrangling (and less of an actual machine learning) project.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make unpack_data` or `make train`
    ├── README.md          <- The top-level README providing an overview of the project.
    ├── data
    │   ├── raw            <- The original, immutable data.
    │   ├── unpacked       <- The data after unpacking json columns
    │   ├── recoded        <- The data after feature recoding and generation.
    │   ├── final          <- The final features (X) and target variable (y) for time windows.
    │   └── final_concat   <- The final concatenated features (X) and target variable (y).
    │
    │
    ├── models             
    │   ├── trained        <- Trained models
    │   └── predictions    <- Model predictions (final submission file)
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to unpack data recode existing and generate new features
    │   │   ├── unpack_dataset.py
    │   │   └── recode_dataset.py
    │   │      
    │   ├── features       <- Scripts to build features and target variables for time windows
    │   │   ├── create_X_y_roll_windows.py
    │   │   └── concat_X_y.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

Workflow 
------------
The workflow is described for Linux, to run it on Windows you will have to install the **make** tool for Windows and get it to work.

- Setup
1. Clone or download this repo.
1. Make sure the **make** tool is installed on your system.
1. Run `make create_environment`, which will install virtualenvwrapper (if not already installed) and create a new virtual environment called google-analytics-challenge. If conda is installed, it will be used to create a new environment. Alternatively, you can create a new virtual environment yourself using a different tool.
1. Activate the new environment. If you use virtualenvwrapper, run `workon google-analytics-challenge`.
1. Run `make requirements` to install the required packages.
1. Download competition data from Kaggle and put the files train_v2.csv, test_v2.csv, sample_submission_v2.csv in the folder data/raw. 

- Analysis scripts
1. Run `make unpack_data` to run the script that unpacks columns containing json fields and drops unnecessary columns (this will take several hours due to the size of the data).
1. Run `make recode_data` to run the script that recodes some features and generates some new.
1. Run `make create_X_y` to run the script that creates time window based aggregated features and corresponding target variable values.
1. Run `make concat_X_y` to run the script that concatenates time window based aggregated features and target variable values into one dataframe.
1. Run `make train` to run the script that trains the model using the concatenated data, makes prediction for the prediction time period, stores the trained model and the predictions in the format ready for submission on Kaggle.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
