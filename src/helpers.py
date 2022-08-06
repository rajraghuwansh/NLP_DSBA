import torch
import numpy as np
import pandas as pd

COLUMN_NAMES = ["sentiment", "aspect_category", "aspect_term", "position", "sentence"]
CATEGORY_NAMES = ['AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY',
                  'DRINKS#STYLE_OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
                  'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL',
                  'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL']
NUM_CATEGORIES = 12
MAX_LENGTH = 60
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 25
EMBEDDING_SIZE = 768
TRAIN_SAMPLES = 1503
VALID_SAMPLES = 376
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_target_and_category(df):
    df["target"] = np.nan
    df.loc[df.sentiment == "positive", "target"] = 0
    df.loc[df.sentiment == "negative", "target"] = 1
    df.loc[df.sentiment == "neutral", "target"] = 2
    df_new = pd.get_dummies(df.aspect_category)
    existing_columns = set(df_new.columns)
    for column in CATEGORY_NAMES:
        if column in existing_columns:
            df[column] = df_new[column]
        else:
            df[column] = 0

    df["aspect_category"] = df["aspect_category"].str.lower().str.replace("#", "-")
    df["sentence"] = df["aspect_category"] + "-" + df["aspect_term"] + ": " + df["sentence"]
    return df

