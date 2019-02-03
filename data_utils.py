import pandas as pd


def read_multiple_csv(path="./data/trip"):
    import glob
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)

    frame = pd.concat(list_)
    return frame


def check_df_na(df):
    na_cnt = sum(df.isnull().sum())
    if na_cnt != 0:
        raise Exception("find na in dataframe")
