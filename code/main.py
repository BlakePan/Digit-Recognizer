#flow:
#step0 define functions and perceptron class
#step1 read data from files (train.csv and test.csv)
#    need information:
#    y -> lable
#    X -> input vector
#    d -> dimension of X
#    N -> number of data
#step2 resturct data to feasible format
#    yn = g(Xnj), j=1~d
#    target: find g !
#step3 feed data for training proceptron
#step4 evaluate g(X) by test data
#step final close files

import numpy as np
from numpy import linalg as LA
import csv
import logging
import random

#logging setting
log_file = "./LOG.log"
log_level = logging.INFO

logger = logging.getLogger("main")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

logging.disable(logging.CRITICAL)

#Start
#step0 define functions and perceptron class


class Perceptron:
    def __init__(self, Pid, d):
        self.Pid = Pid  # perceptron id
        self.dim = d
        self.y = {'ID': 0, 'ACT': 0, 'RESPONSE': 0}
        #self.w = np.ones(self.dim)
        self.w = []
        for i in range(self.dim):
            self.w.append(random.uniform(-1, 1))
        #self.th = random.uniform(0, 255)
        self.th = 50

        logger.debug("Perceptron id%d" % self.Pid)
        logger.debug("Thresold: %f" % self.th)
        logger.debug("Weight: ")
        logger.debug(self.w)

    def UpdateWeight(self, X, y):
        X = np.array(X, dtype=float)
        X = X / LA.norm(X)    # Normalization
        self.w = self.w + y * X

    def CalResult(self, X):
        self.y['ID'] = self.Pid
        tmp = np.dot(self.w, X) - self.th
        self.y['RESPONSE'] = tmp
        if tmp > 0:
            self.y['ACT'] = 1
        else:
            self.y['ACT'] = -1

    def GetWeight(self):
        return self.w

    def GetThershold(self):
        return self.th

    def GetResult(self):
        return self.y


def Train(P, X, desire_y):
    P.CalResult(X)
    tmp_y = P.GetResult()
    logger.info("Perceptron NO.%d" % tmp_y['ID'])
    logger.info("Response: %d" % tmp_y['RESPONSE'])
    logger.info("Active: %d" % tmp_y['ACT'])
    if desire_y != tmp_y['ACT']:
        logger.info("Mistake! update weight")
        P.UpdateWeight(X, desire_y)


Class = 10    # multi-classfication
Iteration = 100

if __name__ == "__main__":

#step1 read data from files (train.csv and test.csv)
    logger.info("Step1 read data")
    f1 = open('../train.csv', "r")
    f2 = open('../test.csv', "r")
    #f2 = open('../pseudotest.csv', "r")
    train_data = csv.reader(f1)
    test_data = csv.reader(f2)
    logger.info("read data finish")

#step2 resturct data to feasible format
    logger.info("Step2 resturct data")
    Labels = []
    X = []    # input vector
    read_index = 0
    for row in train_data:
        if read_index:
            Labels.append(row[0])
            X.append(np.array(row[1:]))
        read_index = read_index + 1

    #Labels = np.array(Labels, dtype=float)
    d = len(X[0])    # d -> dimension of X
    N = len(X)       # N -> number of data
    '''
    #Normalization
    X = np.array(X, dtype=float)
    X = X / LA.norm(X)
    '''
    logger.info("resturct data finish")
    logger.debug(N)
    logger.debug(d)
    logger.debug(len(X))
    logger.debug((X[0][132]))
    logger.debug((X[14][125]))
    logger.debug((X[54][129]))
    logger.debug((X[16][514]))
    logger.debug((X[41999][783]))
    logger.debug((X[0][0:]))

#step3 feed data for training proceptron
#create 10 proceptrons for multiclassification
    logger.info("Step3 training")
    Perceptron_list = []
    cur_x = np.array([])
    for i in range(Class):
        Perceptron_list.append(Perceptron(i, d))

    for i in range(N):
        print(("Training %f%%" % (float(i) / N * 100)))
        logger.info(("Data NO%d" % (i)))
        cur_label = int(Labels[i])    # current desired label
        logger.info(("cur_label = %d" % cur_label))
        cur_x = X[i]             # current input vector
        cur_x = np.array(cur_x, dtype=float)
        for j in range(Class):
            logger.info(("Train Per NO.%d" % j))
            if j == cur_label:    # perceptron NO y should say 1
                logger.info("active")
                Train(Perceptron_list[j], cur_x, 1)
            else:                 # others should say -1
                logger.info("in-active")
                Train(Perceptron_list[j], cur_x, -1)

    logger.info("training finish")

    for i in range(Class):
        logger.debug(("Perceptron NO%d" % i))
        logger.debug((Perceptron_list[i].GetWeight()))

#step4 evaluate g(X) by test data
    logger.info("Step4 test data")
    T = []    # test data
    read_index = 0
    for row in test_data:
        if read_index:
            T.append(np.array(row[0:]))
        read_index += 1

    N_T = len(T)       # number of test data
    Record = []        # record for test result
    New_Record = []    # record after delete in-active Perceptron

    for i in range(N_T):
        print(("Testing %f%%" % (float(i) / N_T * 100)))
        logger.info("Test data NO.%d" % i)
        cur_t = T[i]             # current test vector
        cur_t = np.array(cur_t, dtype=float)
        logger.info("cur test data:")
        logger.info(cur_t)
        for j in range(Class):
            logger.info(("Test Perceptron NO.%d" % j))
            Perceptron_list[j].CalResult(cur_t)
            Record.append(Perceptron_list[j].GetResult())
        logger.info("Record:")
        logger.info(Record)

        # keep active perceptrons and find the one with max response
        maxtmp = 0
        act_index = 0
        for i in range(Class):
            if Record[i]['ACT'] == 1:
                if Record[i]['RESPONSE'] > maxtmp:
                    maxtmp = Record[i]['RESPONSE']
                    act_index = i

        if act_index < 0:
            New_Record.append({'ID': -1, 'ACT': 0, 'RESPONSE': 0})
        else:
            New_Record.append(Record[act_index])
        logger.info("New Record:")
        logger.info(New_Record)

    logger.info("test data finish")

#step5 write result to csv file
    logger.info("Step5 write data")

    with open('PLA.csv', 'w') as f3:
        csv_writer = csv.writer(f3)
        csv_writer.writerow(["ImageId", "Label"])
        for y in range(N_T):
            csv_writer.writerow([y + 1, New_Record[y]['ID']])

    logger.info("write data finish")

#step final close files
    f1.close()
    f2.close()
    logger.info("End of program")
