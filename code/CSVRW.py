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
    #open files
    f1 = open('../train.csv', "r")
    f2 = open('../test.csv', "r")
    #f2 = open('../pseudotest.csv', "r")
    train_data = csv.reader(f1)
    test_data = csv.reader(f2)

    #Labels = []
    #read training data and lables
    #X = []    # input vector
    read_index = 0
    for row in train_data:
        if read_index:
            Labels.append(row[0])
            X.append(np.array(row[1:]))
        read_index = read_index + 1

    #read testing data
    #T = []    # test data
    read_index = 0
    for row in test_data:
        if read_index:
            T.append(np.array(row[0:]))
        read_index += 1

    #close files
    f1.close()
    f2.close()

if __name__ == "__main__":
    Labels = []
    X = []
    T = []
    CSV_read(Labels, X, T)

    logger.info("in main")
    logger.info("Labels:")
    logger.info(Labels)
    logger.info("X:")
    logger.info(X)
    logger.info("T:")
    logger.info(T)