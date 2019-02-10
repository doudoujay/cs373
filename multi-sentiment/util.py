import pandas as pd
import numpy as np
import csv

def main():
    trainData = pd.read_csv("Headline_Trainingdata.csv")
    trainData = trainData.drop(trainData.columns[0],axis=1)
    trainData['sentiment'] = trainData['sentiment'].astype(str)
    cols = trainData.columns.tolist()
    cols = cols[1:] + cols[:1]
    trainData = trainData[cols]
    trainData.to_csv("training.csv", index=False,quoting=csv.QUOTE_NONNUMERIC, header=False)


if __name__ == "__main__":
    main()
    