# Bike sharing forecast
Bike sharing forecast per site

# Instruction 
This project is for group study of Chicago Bike sharing forecast per site.
Utilizing XGBoost and Deep Learning.

# Files
- clustering.py: Clusering process using AgglomerativeClustering
- config.py: Project basic configuration
- data_cleaning.py: Data cleansing and feature engineering
- data_utils.py: Common utils like loading csv files and check na in dataframe
- df_utils.py: Pandas dataframe utils
- ml_main.py: The entry function main():
- ml_pipeline.py: Machine learning pipeline. Currently only for XGBoost
- unit test for clustering.py

# Initialization
1. git pull/download this project
2. download data fold and put into the root folder of this project
    The data folder should contains station, trip, weather sub folders
3. run by ml_main()
4. Do change forecast_end_date, process_days to get forecast multiple days to get average R2 score.

