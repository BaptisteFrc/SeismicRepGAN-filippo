import glob
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py



def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def read_th_engine(filename):
    # reads each csv file to a pandas.DataFrame
    df_csv = pd.read_csv(filename, header=None,
                         names=["ACC{:>s}".format(filename.split("Acc_")[1].strip(".csv"))],
                         dtype={"ACC{:>s}".format(filename.split("Acc_")[1].strip(".csv")): np.float64},)
    return df_csv

nX,nXchannels,Xsize=2000,4,2048
signal=2
rawdata_dir=["PortiqueElasPlas_N_2000_index", "PortiqueElasPlas_E_2000_index"]
store_dir="input_data"
latentCdim=2
nX2,pas_de_t=500, 6 #eviter plus sinon perte d'info
Xsize2=Xsize//pas_de_t
reste=Xsize%pas_de_t

def CreateData():
    # CreateData.__globals__.update(kwargs)

    data = np.zeros((nX2*signal,nXchannels,Xsize2),dtype=np.float32)
    
    for i in range(signal):

        fid_list_th = sorted(glob.glob(opj(rawdata_dir[i],"Acc_*.csv")))
        th_channels = [int(fname.split("Acc_")[1].strip(".csv")) for fname in fid_list_th]
        
        # load time histories
        df_th = pd.concat([read_th_engine(fname) for fname in fid_list_th],axis=1)
        
        # load the damage index
        # df_di = pd.read_csv(opj(rawdata_dir[i], "damage_index.csv"), header=None,
        #                     names=["DI"], dtype={"DI": np.float64},)
        # damage_index = df_di.to_numpy()[:nX]
        
        # max_nX = df_di.shape[0]
        # assert max_nX>nX
        # step = nX//signal
        # max_step = max_nX//signal
        
        for s in range(signal):
            # iBeg = s*max_step*Xsize
            # iEnd = iBeg+step*Xsize-1
            for c in th_channels:
                a=df_th.loc[:nX2*Xsize-1, "ACC{:>d}".format(c)].to_numpy().astype(np.float32).reshape(nX2,Xsize)
                if reste==0 :
                    b=a[:,::pas_de_t]
                else :
                    b=a[:,:-reste:pas_de_t]
                data[s*nX2:(s+1)*nX2, c-1, :] = b

    pga = np.tile(np.atleast_3d(np.max(data,axis=-1)),(1,1,Xsize2))
    
    X = data/pga
    
    X = np.pad(X,((0,0),(0,0),(0,nxtpow2(X.shape[-1])-X.shape[-1])))
    X = np.swapaxes(X,1,2)

    # n = []
    # fid_th = open(opj(store_dir,"magnitude.csv"))
    # th = csv.reader(fid_th)
    # for row in th:
    #     n.append(row)

    # m = np.array(n,dtype=np.float32)
    # magnitude = np.zeros((nX,1),dtype=np.float32)
    # for i in range(int(nX/N/signal)):
    #     magnitude[i+int(nX/signal)] = m[i]
    #     magnitude[i+int(nX/signal)+int(nX/signal/N)] = m[i]

    # damage_class = np.zeros((nX,latentCdim),dtype=np.float32)
   
    # for i in range(latentCdim):
    #     damage_class[nX//latentCdim*i:nX//latentCdim*(i+1),i] = 1.0

    # for i in range(latentCdim):
    #     h5f = h5py.File(opj(store_dir,"Damaged_{:>d}.h5".format(i)),'w')
    #     h5f.create_dataset('X{:>d}'.format(i), data=X[nX//latentCdim*i:nX//latentCdim*(i+1),:,:])
    #     h5f.create_dataset('c{:>d}'.format(i), data=damage_class[nX//latentCdim*i:nX//latentCdim*(i+1),:])
    #     h5f.create_dataset('magnitude{:>d}'.format(i), data=magnitude[nX//latentCdim*i:nX//latentCdim*(i+1),:])
    #     h5f.create_dataset('d{:>d}'.format(i), data=damage_index[nX//latentCdim*i:nX//latentCdim*(i+1)])
    #     h5f.close()

    h5f = h5py.File(opj(store_dir,"Data2.h5"),'w')
    h5f.create_dataset('X', data=X)
    # h5f.create_dataset('damage_class', data=damage_class)
    # h5f.create_dataset('magnitude', data=magnitude)
    # h5f.create_dataset('damage_index', data=damage_index)
    h5f.close()

    X = shuffle(X, random_state=0)

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld = train_test_split(X, random_state=0, test_size=0.1)
    
    # return (
    #     tf.data.Dataset.from_tensor_slices((Xtrn,(Ctrn,Mtrn,Dtrn))).batch(batchSize),
    #     tf.data.Dataset.from_tensor_slices((Xvld,(Cvld,Mvld,Dvld))).batch(batchSize)
    #     )

    return Xtrn, Xvld

CreateData()

def LoadData():
    # LoadData.__globals__.update(kwargs)

    fid_th = opj(store_dir,"Data2.h5")
    
    h5f = h5py.File(fid_th,'r')
    X = h5f['X'][...]
    # c = h5f['damage_class'][...]
    # magnitude = h5f['magnitude'][...]
    # d = h5f['damage_index'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld = train_test_split(X, random_state=0, test_size=0.1)
    
    # return (
    #     tf.data.Dataset.from_tensor_slices((Xtrn,(Ctrn,Mtrn,Dtrn))).batch(batchSize),
    #     tf.data.Dataset.from_tensor_slices((Xvld,(Cvld,Mvld,Dvld))).batch(batchSize)
    #     )

    return Xtrn, Xvld

# LoadData()



def Load_Un_Damaged(i,**kwargs):
    Load_Un_Damaged.__globals__.update(kwargs)

    fid_th = opj(store_dir,"Damaged_{:>d}.h5".format(i))
    h5f = h5py.File(fid_th,'r')
    X = h5f['X{:>d}'.format(i)][...]
    c = h5f['c{:>d}'.format(i)][...]
    magnitude = h5f['magnitude{:>d}'.format(i)][...]
    d = h5f['d{:>d}'.format(i)][...]

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld, Ctrn, Cvld, Mtrn, Mvld, Dtrn, Dvld = train_test_split(X, c, magnitude, d,
                                                                      random_state=0,
                                                                      test_size=0.1)

    return (
        tf.data.Dataset.from_tensor_slices((Xtrn,Ctrn,Mtrn,Dtrn)).batch(batchSize),
        tf.data.Dataset.from_tensor_slices((Xvld,Cvld,Mvld,Dvld)).batch(batchSize)
        )
