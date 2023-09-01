import argparse
import serial
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

def main():
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    timeList = []
    port = args.com  # set port number
    ser = serial.Serial(port=port, baudrate=args.baudrate)  # open the serial
    print(ser)
    ofp = open(file='log/res_{}.txt'.format(t), mode='w')  # make a new log file

    # Extract subject ID, filename, and label
    subject_data = {}
    with open('./test_indice.txt', 'r') as indice_file:
        for line in indice_file:
            label, filename = line.strip().split(',')
            subject_id = filename.split('-')[0]
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append((filename, label))

    all_metrics = []
    subjects_above_threshold = 0
    total_subjects = len(subject_data)

    # Perform calculations for each participant
    for subject_idx, (subject_id, file_info_list) in enumerate(subject_data.items(), start=1):
        y_true_subject = []
        y_pred_subject = []
        subject_desc = f'Subject {subject_id}:'
        file_tqdm = tqdm(file_info_list, desc=subject_desc, leave=True)

        for file_info in file_tqdm:
            filename, true_label = file_info
            y_true_subject.append(true_label)
            # load data from txt files and reshape to (1, 1, 1250, 1)
            testX = txt_to_numpy(args.path_data + filename, 1250).reshape(1, 1, 1250, 1)
            for i in range(0, testX.shape[0]):
                # don't continue running the code until a "begin" is received, otherwise receive iteratively
                while ser.in_waiting < 5:
                    pass
                    time.sleep(0.01)

                # when receiving the code "begin", send the test data cyclically
                recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                # clear the input buffer
                ser.reset_input_buffer()
                if recv.strip() == 'begin':
                    for j in range(0, testX.shape[1]):
                        for k in range(0, testX.shape[2]):
                            for l in range(0, testX.shape[3]):
                                send_str = str(testX[i][j][k][l]) + ' '
                                ser.write(send_str.encode(encoding='utf8'))

                    # Set a time point to represent that all data are sent and the development board starts to perform model inference.
                    start_time = datetime.now()
                    # don't continue running the code until a "ok" is received
                    while ser.in_waiting < 2:
                        pass
                    time.sleep(0.01)
                    recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                    ser.reset_input_buffer()
                    if recv.strip() == 'ok':
                        time.sleep(0.02)
                        # send status 200 to the board
                        send_str = '200 '
                        ser.write(send_str.encode(encoding='utf8'))
                        time.sleep(0.01)
                    # receive results from the board, which is a string separated by commas
                    while ser.in_waiting < 1:
                        pass
                    recv = ser.read(size=1).decode(encoding='utf8')
                    ser.reset_input_buffer()
                    end_time = datetime.now()
                    results = recv.strip()
                    if results == '0':
                        y_pred_subject.append('0')
                    else:
                        y_pred_subject.append('1')
                    # the total time minus sleep/halt time on PC and MCU
                    timeList.append(
                        ((end_time - start_time).seconds * 1000) + ((end_time - start_time).microseconds / 1000) - 44)
                    ofp.write(str(results) + '\r')

        C = confusion_matrix(y_true_subject, y_pred_subject)
        if C.shape == (2, 2):
            acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])

            if (C[1][1] + C[0][1]) != 0:
                precision = C[1][1] / (C[1][1] + C[0][1])
            else:
                precision = 0.0

            if (C[1][1] + C[1][0]) != 0:
                sensitivity = C[1][1] / (C[1][1] + C[1][0])
            else:
                sensitivity = 0.0

            FP_rate = C[0][1] / (C[0][1] + C[0][0])

            if (C[1][1] + C[1][0]) != 0:
                PPV = C[1][1] / (C[1][1] + C[1][0])
            else:
                PPV = 0.0

            NPV = C[0][0] / (C[0][0] + C[0][1])

            if (precision + sensitivity) != 0:
                F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
            else:
                F1_score = 0.0

            if ((2 ** 2) * precision + sensitivity) != 0:
                F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)
            else:
                F_beta_score = 0.0

        all_metrics.append([acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score])
        if F_beta_score > 0.95:
            subjects_above_threshold += 1

        # Update the progress bar
        if subject_idx < total_subjects:
            next_subject_id = list(subject_data.keys())[subject_idx]
            next_subject_desc = f'Files for Subject {next_subject_id}:'
            file_tqdm.set_description(next_subject_desc, refresh=True)
            file_tqdm.leave = True

    ofp.close()
    # Calculate average performance metrics
    total_time = sum(timeList)
    avg_time = np.mean(timeList)
    subject_metrics_array = np.array(all_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score = average_metrics
    print("Final accuracy:", acc)
    print("Final precision:", precision)
    print("Final sensitivity:", sensitivity)
    print("Final FP_rate:", FP_rate)
    print("Final PPV:", PPV)
    print("Final NPV:", NPV)
    print("Final F1_score:", F1_score)
    print("Final F_beta_score:", F_beta_score)
    print("total_time:", total_time)
    print("avg_time:", avg_time)

    proportion_above_threshold = subjects_above_threshold / total_subjects
    g_score = proportion_above_threshold
    print("G Score:", g_score)

    f = open('./log/log_{}.txt'.format(t), 'a')
    f.write("Final Accuracy: {}\n".format(acc))
    f.write("Final Precision: {}\n".format(precision))
    f.write("Final Sensitivity: {}\n".format(sensitivity))
    f.write("Final FP_rate: {}\n".format(FP_rate))
    f.write("Final PPV: {}\n".format(PPV))
    f.write("Final NPV: {}\n".format(NPV))
    f.write("Final F1_Score: {}\n".format(F1_score))
    f.write("Final F_beta_Score: {}\n".format(F_beta_score))
    f.write("G score: {}\n\n".format(g_score))
    f.write("Total_Time: {}\n".format(total_time))
    f.write("Average_Time: {}\n\n".format(avg_time))
    f.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--com', type=str, default='com5')
    argparser.add_argument('--baudrate', type=int, default=115200)
    argparser.add_argument('--path_data', type=str,default='path/to/dataset')
    args = argparser.parse_args()
    main()
