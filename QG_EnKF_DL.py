import sys
import numpy as np
import netCDF4 as nc
from saveNCfile import savenc
from Drymodel import drymodel
#import scipy.io as sio


import keras 
import keras.backend as K
#from data_manager import ClutteredMNIST
#from visualizer import plot_mnist_sample
#from visualizer import print_evaluation
#from visualizer import plot_mnist_grid
import netCDF4
import numpy as np
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
from keras.callbacks import History
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D, merge
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

#from utils import get_initial_weights
#from layers import BilinearInterpolation

__version__ = 0.1






def EnKF(ubi,w,R,B,N,M,state):
     
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations
    n=Nlat*Nlon*2
    m=n
#    combined_state=np.concatenate((ubi,state), axis=1)
#    state_mean = np.mean(combined_state,1)
#    print('state mean', np.shape(state_mean))
    # compute the mean of forecast ensemble
    ub = np.mean(ubi,1)   
    Pb = (1/(N-1)) * (ubi - ub.reshape(-1,1)) @ (ubi - ub.reshape(-1,1)).T
    print('Inside KF, ub',np.shape(ub))
    # compute Jacobian of observation operator at ub
    Dh = np.eye(n,n)
    # compute Kalman gain
    D = Dh@B@Dh.T + R
    
    K = B @ Dh.T @ np.linalg.inv(D)
        
    print('Inside KF, K',np.shape(K))    

    wi = np.zeros([m,M])
    uai = np.zeros([n,M])
    

    for i in range(M):
        # create virtual observations
        wi[:,i] = w + np.random.normal(0,sig_m,[2*Nlat*Nlon,])
        # compute analysis ensemble
        uai[:,i] = state[:,i] + K @ (wi[:,i]-state[:,i])
        
    print('Inside KF, uai',np.shape(uai))
    # compute the mean of analysis ensemble
    ua = np.mean(uai,1)    

    print('Inside KF, ua',np.shape(ua))
    # compute analysis error covariance matrix
    P = (1/(N-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
    return uai, P, Pb


### Define Data-driven architecture ######
def stn(input_shape=(192, 96,2), sampling_size=(8, 16), num_classes=10):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv3)

    conv5 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv5)

    up6 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x)), conv2])
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv6)

    up7 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv7)

### Use tanh in this last layer
    conv10 = Convolution2D(2, (5, 5), activation='linear',padding='same')(conv7)

#    model = Model(input=inputs, output=conv10)
    model = Model(inputs, conv10)

    return model

model = stn()
model.compile(loss='mse', optimizer='adam')
model.summary()

model.load_weights('best_weights_lead1.h5')

### Load dataset for truth and Obs #########


F=nc.Dataset('/ocean/projects/atm170004p/achatto2/Tensorflow/Spatio-temporal/QG_DA/T63/PSI_output.nc')

psi=np.asarray(F['PSI'])
psi=psi[2500:,:,:,:]

#MEAN_L1 = np.mean(psi[:,0,:,:].flatten())
#STD_L1  = np.std(psi[:,0,:,:].flatten())

#MEAN_L2 = np.mean(psi[:,1,:,:].flatten())
#STD_L2  = np.std(psi[:,1,:,:].flatten())


MEAN_L1=0
STD_L1=1
MEAN_L2=0
STD_L2=1


y=np.asarray(F['lat'])
x=np.asarray(F['lon'])

Nlat=np.size(psi,2);
Nlon=np.size(psi,3);

print('size of Nlat',Nlat)
print('size of Nlon',Nlon)

######## Emulate Observation with noise ########

sig_m= 0.15  # standard deviation for measurement noise
R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)

DA_cycles=int(5)
obs=np.zeros([int(np.size(psi,0)/DA_cycles),2,Nlat,Nlon])

obs_count=0
for k in range(DA_cycles,np.size(psi,0),DA_cycles):
  
    obs[obs_count,:,:,:]=psi[k,:,:,:]
   
    obs_count=obs_count+1




obs_tran=np.zeros([np.size(obs,0),Nlon, Nlat, 2])
for k in range(0,np.size(obs,0)):
    obs_l1 = np.transpose(obs[k,0,:,:])
    obs_l2 = np.transpose(obs[k,1,:,:])

    obs_tran[k,:,:,0] = obs_l1
    obs_tran[k,:,:,1] = obs_l2



########### Start initial condition ##########
psi0=psi[0,:,:,:]
psi0l1=np.transpose(psi[0,0,:,:])
psi0l2=np.transpose(psi[0,1,:,:])
psi0=np.zeros([Nlon,Nlat,2])

psi0[:,:,0]=psi0l1
psi0[:,:,1]=psi0l2

psi0_noisy=(psi0.flatten()+np.random.normal(0,sig_m,[2*Nlat*Nlon,])).reshape([Nlon,Nlat,2])



print('shape of initial condition',np.shape(psi0_noisy))

####################


N = int(sys.argv[1])
M=10

print('number of numerical ens',M)
print('number of DD ens',N)

sig_b= 0.1
B = sig_b**2*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
Q = 0.0*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)

###################### Start Assimilation #######
E_tr = np.zeros([Nlat,Nlon,2])
pred_temp_tr = np.zeros([Nlon,Nlat,2])

psi_ensemble=np.zeros([2*Nlat*Nlon,N])
psi_ensemble_new=np.zeros([2*Nlat*Nlon,N])
psi_ensemble_new_denorm=np.zeros([2*Nlat*Nlon,N])


psi_ensemble_numerical=np.zeros([2*Nlat*Nlon,M])
psi_ensemble_numerical_new=np.zeros([2*Nlat*Nlon,M])

T = 300
count=0
psi_updated=np.zeros([T+1,Nlon,Nlat,2])
Pb_updated=np.zeros([Nlat*Nlon*2,Nlat*Nlon*2])
psi_updated_DD =np.zeros([T+1,Nlon,Nlat,2])

t=0
while (t<T+1):
  if (t==0): 

####### Generate ensembles for Unet ################################################################    
    psi0_noisy_L1 = (psi0_noisy[:,:,0]-MEAN_L1)/STD_L1
    psi0_noisy_L2 = (psi0_noisy[:,:,1]-MEAN_L2)/STD_L2
    psi0_noisy_new = np.zeros([Nlon, Nlat, 2])
    psi0_noisy_new[:,:,0]=psi0_noisy_L1
    psi0_noisy_new[:,:,1]=psi0_noisy_L2

    

    for k in range(0,N):
     psi_ensemble[:,k] = (psi0_noisy_new.flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,]))  ### This should be looked at ###
    
####### Generate ensembles for numerical solver ###########################
    
    for k in range (0,M):
     psi_ensemble_numerical[:,k] = (psi0_noisy.flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,])) 
    
######### Evolve ensembles with Unet ###################################################### 
    for k in range(0,N):
     E = psi_ensemble[:,k].reshape([Nlon, Nlat, 2])
     Etr_layer1 = E[:,:,0]
     Etr_layer2 = E[:,:,1]
     E_tr[:,:,0] = np.transpose(Etr_layer1)
     E_tr[:,:,1] = np.transpose(Etr_layer2)

     pred_temp = (model.predict(E_tr.reshape([1, Nlat,Nlon,2]))).reshape([Nlat, Nlon, 2])
     pred_temp_layer1 = pred_temp[:,:,0]
     pred_temp_layer2 = pred_temp[:,:,1]

     pred_temp_tr[:,:,0] = np.transpose(pred_temp_layer1)
     pred_temp_tr[:,:,1] = np.transpose(pred_temp_layer2)

     psi_ensemble_new[:,k] = pred_temp_tr.flatten()
 
   
#     psi_ensemble_new[:,k] = (drymodel(psi_ensemble[:,k].reshape([Nlon,Nlat,2]))).flatten()
#     psi_ensemble_new[:,k] = model.predict(((psi_ensemble[:,k]).reshape([1,Nlon,Nlat,2])-MEAN)/STD).flatten()

########## Evolve numerical ensembles with numerical solver #########################################   
    
    for k in range(0,M):
     psi_ensemble_numerical_new[:,k] = (drymodel(psi_ensemble_numerical[:,k].reshape([Nlon,Nlat,2]))).flatten()
    
    psi_updated[t,:,:,:] = (np.mean(psi_ensemble_numerical_new,1)).reshape([Nlon,Nlat,2]) 

    for k in range(0,N):
     uu = psi_ensemble_new[:,k].reshape([Nlon, Nlat, 2])
     uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1 
     uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
     psi_ensemble_new_denorm[:,k] = uu.flatten()
 
#    psi_updated_DD[t,:,:,:] = (np.mean(psi_ensemble_new_denorm,1)).reshape([Nlon,Nlat,2])    
    
    t=t+1
    
  elif (t>0 and (t+1) % DA_cycles ==0):
    
### Evolve ensembles with Unet##################################
    for k in range(0,N):     
     E = psi_ensemble_new[:,k].reshape([Nlon, Nlat, 2])
     Etr_layer1 = E[:,:,0]
     Etr_layer2 = E[:,:,1]
     E_tr[:,:,0] = np.transpose(Etr_layer1)
     E_tr[:,:,1] = np.transpose(Etr_layer2)

     pred_temp = (model.predict(E_tr.reshape([1, Nlat,Nlon,2]))).reshape([Nlat, Nlon, 2])
     pred_temp_layer1 = pred_temp[:,:,0]
     pred_temp_layer2 = pred_temp[:,:,1]

     pred_temp_tr[:,:,0] = np.transpose(pred_temp_layer1)
     pred_temp_tr[:,:,1] = np.transpose(pred_temp_layer2)

     psi_ensemble_new[:,k] = pred_temp_tr.flatten()
     
#### Evolve numerical ensembles  with numerical solver ####################
    for k in range(0,M):
      psi_ensemble_numerical_new[:,k] = (drymodel(psi_ensemble_numerical_new[:,k].reshape([Nlon,Nlat,2]))).flatten()



#### Start ENKF, Pass determinstic state as well frm NM ################    
    print('Starting KF')
    for k in range(0,N):
     uu = psi_ensemble_new[:,k].reshape([Nlon, Nlat, 2])
     uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
     uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
     psi_ensemble_new_denorm[:,k] = uu.flatten()

    P = (1/(N-1)) * (psi_ensemble_new_denorm - (np.mean(psi_ensemble_new_denorm,1)).reshape(-1,1)) @ (psi_ensemble_new_denorm - (np.mean(psi_ensemble_new_denorm,1)).reshape(-1,1)).T
    psi_ensemble_numerical_new, P, Pb = EnKF(psi_ensemble_new_denorm,obs_tran[count,:,:,:].flatten(),R,P,N,M,psi_ensemble_numerical_new)
    Pb_updated=Pb_updated + Pb
    psi_ensemble_numerical_new = np.asarray(psi_ensemble_numerical_new)
    count=count+1

####### Update with mean of ensembles from ENKF output ##############################
    psi_updated[t,:,:,:]=(np.mean(psi_ensemble_numerical_new,1)).reshape([Nlon,Nlat,2])
    
#    psi_updated_DD[t,:,:,:] = (np.mean(psi_ensemble_new_denorm,1)).reshape([Nlon,Nlat,2])   


### Restart DD ensembles based on new numerical update #########################################
    for k in range(0,N):
     psi_ensemble_new[:,k] = ((np.mean(psi_ensemble_numerical_new,1)).reshape([Nlon,Nlat,2]).flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,]))  ### This should be looked at ###

 
    t=t+1

  else:

############### Evolve ensembles with Unet ###################################################
    for k in range(0,N):
     E = psi_ensemble_new[:,k].reshape([Nlon, Nlat, 2])
     Etr_layer1 = E[:,:,0]
     Etr_layer2 = E[:,:,1]
     E_tr[:,:,0] = np.transpose(Etr_layer1)
     E_tr[:,:,1] = np.transpose(Etr_layer2)

###### This is the forward prediction step with Unet ######################################
     pred_temp = (model.predict(E_tr.reshape([1, Nlat,Nlon,2]))).reshape([Nlat, Nlon, 2])
###########################################################################################
     pred_temp_layer1 = pred_temp[:,:,0]
     pred_temp_layer2 = pred_temp[:,:,1]

     pred_temp_tr[:,:,0] = np.transpose(pred_temp_layer1)
     pred_temp_tr[:,:,1] = np.transpose(pred_temp_layer2)

     psi_ensemble_new[:,k] = pred_temp_tr.flatten()


    for k in range(0,N):
      uu = psi_ensemble_new[:,k].reshape([Nlon, Nlat, 2])
      uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
      uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
      psi_ensemble_new_denorm[:,k] = uu.flatten()





############### Evolve with numerical solver #####################
    for k in range(0,M):
      psi_ensemble_numerical_new[:,k] = (drymodel(psi_ensemble_numerical_new[:,k].reshape([Nlon,Nlat,2]))).flatten()        

    psi_updated[t,:,:,:] = (np.mean(psi_ensemble_numerical_new,1)).reshape([Nlon,Nlat,2])
    
#    psi_updated_DD[t,:,:,:] = (np.mean(psi_ensemble_new_denorm,1)).reshape([Nlon,Nlat,2])    

    t=t+1  
  
  print('Out of '+str(T+1),t)

Pb_updated=Pb_updated/(count-1)
savenc(psi_updated, x, y, 'Psi_updated_hybridDL_combined_DDrestartM' + str(M) + 'T'+str(T)+'ens'+str(N)+'.nc')
#savenc(psi_updated_DD, x, y, 'Psi_updated_hybridDL_DA_every5dt_combined_DDrestart_M500ens'+str(N)+'.nc')
np.savetxt('analysis_cov_DDrestart_M' +str(M)+'DL' +str(N)+'T'+str(T)+'.csv',Pb_updated,delimiter=',')

#sio.savemat('analysis_cov_ens'+str(N)+'.mat',dict([('Cov',Pb_updated)]))


