import csv
import numpy as np
import logging

#logging setting
log_file = "./CSVRW.log"
log_level = logging.INFO

logger = logging.getLogger("CSVRW")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

#logging.disable(logging.CRITICAL)


def CSV_read(Labels, X, T):
    with open('../train.csv', 'r') as incsv:
        train_data = csv.reader(incsv)    # read training data and lables
        next(train_data)    # skip first row
        for row in train_data:
            Labels.append(row[0])
            X.append(np.array(row[1:]))

    with open('../pseudotest.csv', 'r') as incsv:
        test_data = csv.reader(incsv)    # read testing data
        next(test_data)    # skip first row
        for row in test_data:
            T.append(np.array(row[0:]))


def CSV_write(Fname, Leng, WTdata):
    with open('%s.csv' % Fname, 'w') as outcsv:
        csv_writer = csv.writer(outcsv)
        csv_writer.writerow(["ImageId", "Label"])
        for y in range(Leng):
            csv_writer.writerow([y + 1, WTdata[y]['ID']])

if __name__ == "__main__":
    Labels = []
    X = []
    T = []
    CSV_read(Labels, X, T)

    logger.debug("in main")
    logger.debug("Labels:")
    logger.debug(Labels)
    logger.debug("X:")
    logger.debug(X)
    logger.debug("T:")
    logger.debug(T)