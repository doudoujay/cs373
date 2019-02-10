import pandas as pd
import numpy as np
import csv
import httplib2
from oauth2client.contrib import gce


def main():
    credentials = gce.AppAssertionCredentials(scope='https://www.googleapis.com/auth/devstorage.read_write')
    http = credentials.authorize(httplib2.Http())
    trainData = pd.read_csv("Headline_Testing.csv")

if __name__ == "__main__":
    main()
    
