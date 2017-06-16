import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
import PAA as paa
import time

# Reading data from the text file
def file_to_list(file_name, ts_len):
    f1 = open(file_name,'r')
    file_data = []

    for line in f1:
        line = line.strip()
        ip_val = float(line)
        file_data.append(ip_val)
    f1.close()

    ts_list = []
    for x in range(0, len(file_data), ts_len):
        ts_list.append(file_data[x:x+ts_len])

    return ts_list

def plot_graphs(ref_list, test_list, anomaly_score, predict):
    plt.subplot(411)
    for x in ref_list:
        plt.plot(range(len(x)), x)
    plt.title("Training Signal")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(412)
    for y in test_list:
        plt.plot(range(len(y)), y)
    plt.title("Test Signal")
    plt.xlabel("Time")
    plt.ylabel("Value")
    #plt.xticks([x for x in range(0, len(test_list),100)])

    plt.subplot(413)
    plt.plot(range(len(anomaly_score)),anomaly_score)
    #plt.xticks([x for x in range(0, len(test_list), 100)])
    plt.title("Anomaly Score")
    plt.xlabel("Sequence Number")
    plt.ylabel("Value")
    #plt.ylim([-1.2,1.2])

    plt.subplot(414)
    plt.plot(range(len(predict)),predict)
    plt.ylim([-1.2, 1.2])
    # plt.xticks([x for x in range(0, len(test_list), 100)])
    plt.title("Detected Anomalies")
    plt.xlabel("Sequence Number")
    plt.ylabel("Value")

    plt.show()


def predict (anomaly_score, contamination):
    anomaly_score = np.array(anomaly_score)
    print ("Anomaly_score_shape = {}".format(anomaly_score.shape[0]))
    contamination = float(contamination)
    print ("Contamination = {}".format(contamination))
    threshold = stats.scoreatpercentile(anomaly_score, 100 * (contamination))
    print ("Threshold = {}".format(threshold))

    is_inlier = np.ones(anomaly_score.shape[0], dtype=int)
    is_inlier[anomaly_score <= threshold] = -1
    return is_inlier


# Main script starts
training_ts_list = file_to_list('multivariant_training7_1359.txt', 1359)
testing_ts_list = file_to_list('multivariant_testing7_1584.txt', 1584)
print (len(training_ts_list[0]), len(training_ts_list))
print (len(testing_ts_list[0]), len(testing_ts_list))

w = 30
m = 10
st = time.time()
training_paa = paa.ts_to_PAA(w,m,training_ts_list)
testing_paa = paa.ts_to_PAA(w,m,testing_ts_list)
print ("PAA time = {}".format(time.time() - st))
#print (training_paa)
print (training_paa.shape)
t1 = time.time()
IF1 = IsolationForest(max_samples= 256, n_estimators=100, contamination= 0.01)
IF1.fit(training_paa)
#cont = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
cont = [0.01, 0.02,0.04,0.08,0.1, 0.2,0.4, 0.5]
for val in cont:

    anomaly_score = IF1.decision_function(testing_paa)
    anomaly_score = [0 for x in range(w)] + [z for z in anomaly_score]
    predict_score = predict(anomaly_score, val)
    predict_socre =  [z for z in predict_score]
    plot_graphs(training_ts_list, testing_ts_list, anomaly_score, predict_score)

    plt.subplot(411)
    plt.title('Training Signal')
    plt.xlabel('Instance Number')
    plt.ylabel('Value')
    plt.plot(range(len(training_ts_list[0])), training_ts_list[0], color='b')
    plt.plot(range(len(training_ts_list[1])), training_ts_list[1], color='r')
    plt.plot(range(len(training_ts_list[2])), training_ts_list[2], color='g')

    plt.subplot(412)
    plt.title('Testing Signal')
    plt.xlabel('Instance Number')
    plt.ylabel('Value')
    plt.plot(range(len(testing_ts_list[0])), testing_ts_list[0], color='b')
    plt.plot(range(len(testing_ts_list[1])), testing_ts_list[1], color='r')
    plt.plot(range(len(testing_ts_list[2])), testing_ts_list[2], color='g')

    plt.subplot(413)
    plt.title('Anomaly Score')
    plt.xlabel('Instance Number')
    plt.ylabel('Anomaly Score')
    plt.plot(range(len(anomaly_score)), anomaly_score)

    plt.subplot(414)
    plt.plot(range(len(predict_score)), predict_score)
    plt.ylim([-1.2,1.2])

    plt.show()
    plt.close()



'''
    # Plots for multivariant series paper results
    plt.subplot(411)
    plt.plot(range(len(testing_ts_list[0])), testing_ts_list[0])
    plt.title('Signal 1')

    plt.subplot(412)
    plt.plot(range(len(testing_ts_list[1])), testing_ts_list[1])
    plt.title('Signal 2')

    #plt.subplot(513)
    #plt.plot(range(len(testing_ts_list[2])), testing_ts_list[2])
    #plt.title('Signal 3')

    plt.subplot(413)
    plt.plot(range(len(anomaly_score)), anomaly_score)
    plt.title('Anomaly Score')

    plt.subplot(414)
    plt.plot(range(len(predict_score)), predict_score)
    plt.title('Predict')

    plt.show()
'''
'''
# Plots for paper results
plt.subplot(511)
plt.plot(range(len(testing_ts_list[0])),testing_ts_list[0])
plt.title('Signal 1')

plt.subplot(512)
plt.plot(range(len(testing_ts_list[1])),testing_ts_list[1])
plt.title('Signal 2')

plt.subplot(513)
plt.plot(range(len(testing_ts_list[2])),testing_ts_list[2])
plt.title('Signal 3')

plt.subplot(514)
plt.plot(range(len(anomaly_score)),anomaly_score)
plt.title('Anomaly Score')

plt.subplot(515)
plt.plot(range(len(predict)),predict)
plt.title('Anomaly Score')

plt.show()
'''