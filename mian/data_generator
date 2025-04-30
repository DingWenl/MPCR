from random import sample
import random
import numpy as np
from keras.utils import np_utils
# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        target_list = list(range(40))
        # setting the K_max to 5 -1 = 4
        K_max = 5
        for i in range(int(batchsize)):
            # randomly selecting a block
            k = sample(train_list, 1)[0]
            # randomly selecting a trial
            m = sample(target_list, 1)[0]
            
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
            time_start = random.randint(35+125,int(1250+35+125-win_train))
            time_end = time_start + win_train
            # get three sub-inputs
            x_11 = train_data1[k][m][:,time_start:time_end]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            
            x_12 = train_data2[k][m][:,time_start:time_end]
            x_22 = np.reshape(x_12,(channel, win_train, 1))

            x_13 = train_data3[k][m][:,time_start:time_end]
            x_23 = np.reshape(x_13,(channel, win_train, 1))
            # concatenate the three sub-input into one input to make it can be as the input of the CNN-Former's network
            x_s = np.concatenate((x_21, x_22, x_23), axis=-1)
            # x_s[:,r_m:int(r_m+int(mask_rate*win_train)),:] = 0
            x_train[i] = x_s
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
                
        x_train = np.array(x_train)
        y_train1 = np.reshape(y_train,(batchsize,40))

        ## MPCR
        recon_list = list(range(batchsize))
        for i in range(batchsize):
            # reshape the sample to a two-dimensional form
            x_data = np.reshape(x_train[i],(channel, win_train*3))
            # abtaining the channel-wise mean and standard deviation
            x_data_mean = np.mean(x_data, axis=1)
            x_data_mean = np.reshape(x_data_mean,(channel, 1))
            x_data_std = np.std(x_data,axis=1)
            x_data_std = np.reshape(x_data_std,(channel,1))
            # obtaining the normalized sample
            x_data_norm = (x_data-x_data_mean)/x_data_std
            # obtaining the covariance matrix
            x_cov = np.dot(x_data_norm,x_data_norm.T)/(win_train*3-1)
            # eigenvalues and eigenvectors calculation
            _, featVec1=np.linalg.eig(x_cov)
            # principal component representation calculation
            x_pcr = np.dot(featVec1.T,x_data_norm) #np.linalg.inv(featVec1),featVec1.T
            # randomly selecting a k value (n_channel) from the range [1, 4]
            n_channel = sample(list(range(1,K_max)), 1)[0]
            # randomly selecting n_channel channels from the 9 channels of a training sample
            channel_n = sample(list(range(9)), n_channel)#[0]
            # masked principal component representation calculation
            x_pcr[channel_n,:] = 0
            x_mpcr = x_pcr
            # data reconstruction
            recon_x = np.dot(featVec1,x_mpcr)*x_data_std + x_data_mean
            recon_x = np.reshape(recon_x,(channel, win_train,3))

            recon_list[i] = recon_x
        x_train1 = np.array(recon_list)
        
        yield x_train1, y_train1




