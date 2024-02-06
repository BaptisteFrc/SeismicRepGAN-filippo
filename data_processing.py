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


    h5f = h5py.File(opj(store_dir,"Data2.h5"),'w')
    h5f.create_dataset('X', data=X)
    h5f.close()

    X = shuffle(X, random_state=0)

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld = train_test_split(X, random_state=0, test_size=0.1)


    return Xtrn, Xvld

# CreateData()

def LoadData():
    # LoadData.__globals__.update(kwargs)

    fid_th = opj(store_dir,"Data2.h5")
    
    h5f = h5py.File(fid_th,'r')
    X = h5f['X'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld = train_test_split(X, random_state=0, test_size=0.1)
    

    return Xtrn, Xvld

# LoadData()

