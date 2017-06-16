import numpy as np
from math import floor


def ts_to_PAA (w, m, input_ts):
    m = float (m)
    N = len(input_ts[0])
    #n = N - w +1

    for i in range(len(input_ts)):
        lt1 = input_ts[i] #Copy ith variate time series to lt1
        lt2 = [lt1[j:j+w] for j in range(len(lt1)-w+1)]
        arr2 = np.array(lt2)
        arr2.resize(int((len(lt1)-w+1) * m), int(floor(w/m)))
        arr3 = np.ones((int(floor(w/m)),1))
        arr4 = (m/w) * np.dot(arr2,arr3)
        arr5 = np.reshape(arr4,(len(lt1)-w+1, int (m)))

        if(i == 0):
            arr6 = arr5
        else:
            arr6 = np.concatenate((arr6,arr5), axis=1)

    return arr6

# For Univariate
def ts_to_norm_PAA (w ,m, input_ts):
    m = float(m)
    N = len(input_ts[0])
    # n = N - w +1

    for i in range(len(input_ts)):
        lt1 = input_ts[i]  # Copy ith variate time series to lt1
        lt2 = [lt1[j:j + w] for j in range(len(lt1) - w + 1)]
        # Normalization
        #arr1 = np.asanyarray(lt1)
        arr1 = np.asanyarray(lt2)
        amean = (arr1.mean(axis=1)).reshape(-1, 1)
        astd = (arr1.std(axis=1)).reshape(-1, 1)

        arr2 = (arr1 - amean)/ astd
        arr2.resize(int((len(lt1) - w + 1) * m), int(floor(w / m)))
        arr3 = np.ones((int(floor(w / m)), 1))
        arr4 = (m / w) * np.dot(arr2, arr3)
        arr5 = np.reshape(arr4, (len(lt1) - w + 1, int(m)))

        if (i == 0):
            arr6 = np.concatenate((arr5,amean,astd),axis=1)
        else:
            arr6 = np.concatenate((arr6, arr5, amean, astd), axis=1)

    return arr6

# For Multivariate
def ts_to_norm_PAA1 (w ,m, input_ts):
    m = float(m)
    N = len(input_ts[0])
    # n = N - w +1

    for i in range(len(input_ts)):
        lt1 = input_ts[i]  # Copy ith variate time series to lt1
        lt2 = [lt1[j:j + w] for j in range(len(lt1) - w + 1)]
        # Normalization
        #arr1 = np.asanyarray(lt1)
        arr1 = np.asanyarray(lt2)
        amean = (arr1.mean(axis=1)).reshape(-1, 1)
        astd = (arr1.std(axis=1)).reshape(-1, 1)

        arr2 = (arr1 - amean)/ astd
        arr2.resize(int((len(lt1) - w + 1) * m), int(floor(w / m)))
        arr3 = np.ones((int(floor(w / m)), 1))
        arr4 = (m / w) * np.dot(arr2, arr3)
        arr5 = np.reshape(arr4, (len(lt1) - w + 1, int(m)))

        if (i == 0):
            arr6 = arr5
        else:
            arr6 = np.concatenate((arr6, arr5),axis = 1)

    return arr6


# Normalize whole time series once not the individual sub-sequence
def ts_to_norm_PAA2 (w ,m, input_ts):
    m = float(m)
    N = len(input_ts[0])
    # n = N - w +1

    for i in range(len(input_ts)):
        lt1 = input_ts[i]  # Copy ith variate time series to lt1
        arr1 = np.asanyarray(lt1)
        amean = (arr1.mean())
        astd =  arr1.std()
        arr2 = (arr1 - amean) / astd
        lt11 = arr2.tolist()
        lt2 = [lt11[j:j + w] for j in range(len(lt1) - w + 1)]
        # Normalization
        #arr1 = np.asanyarray(lt1)

        arr2 = np.asanyarray(lt2)
        arr2.resize(int((len(lt1) - w + 1) * m), int(floor(w / m)))
        arr3 = np.ones((int(floor(w / m)), 1))
        arr4 = (m / w) * np.dot(arr2, arr3)
        arr5 = np.reshape(arr4, (len(lt1) - w + 1, int(m)))

        if (i == 0):
            arr6 = arr5
        else:
            arr6 = np.concatenate((arr6, arr5),axis = 1)

    return arr6