import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import sparse
import cv2
from sklearn.neighbors import KernelDensity




def calAttenMap(filename):
    finalmask=np.zeros((8000,8000))
    counts=0
    cells_x=[]
    cells_y=[]
    cellski_x=[]
    cellski_y=[]
    cellsphh_x=[]
    cellsphh_y=[]


    item='./colorDeconv/{}_ki_deconv.jpg'.format(filename)
    kide=np.array(Image.open(item))
    kide=kide[:,:,0].reshape(8000,8000)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    kide = cv2.dilate(kide, kernel, iterations=2)
    kide[kide>0]=1

    phhde=np.array(Image.open(item.replace('ki','phh')))
    phhde=phhde[:,:,0].reshape(8000,8000)
    phhde = cv2.dilate(phhde, kernel, iterations=2)
    phhde[phhde>0]=1


    for i in range(8):
        for j in range(8):
            name=filename+'_{:0>2}{:0>2}'.format(i,j)        
            masks=sparse.load_npz('./detectResults/{}.npz'.format(name))
            masks=masks.todense()
            if masks.size==0:
                    continue

            subki=kide[i*1000:i*1000+1000, j*1000:j*1000+1000]
            subphh=phhde[i*1000:i*1000+1000, j*1000:j*1000+1000]

            counts+=masks.shape[-1]
            #finalmask[i*1000:i*1000+1000, j*1000:j*1000+1000]=np.sum(masks,axis=-1)
            xf=i*1000
            yf=j*1000
            for k in range(masks.shape[-1]):
                mask=masks[:,:,k]
                x,y=np.where(mask==1)
                cells_x.append(np.mean(x)+xf)
                cells_y.append(np.mean(y)+yf)
                if subki[int(cells_x[-1]),int(cells_y[-1])]>0:
                    cellski_x.append(np.mean(x)+xf)
                    cellski_y.append(np.mean(y)+yf)
                if subphh[int(cells_x[-1]),int(cells_y[-1])]>0:
                    cellsphh_x.append(np.mean(x)+xf)
                    cellsphh_y.append(np.mean(y)+yf)


    if not cellski_y+cellsphh_y:
        return finalmask[0:800]
    xy_train  = np.vstack([cellski_y+cellsphh_y, cellski_x+cellsphh_x]).T
    xx, yy = np.mgrid[0:8000:80,0:8000:80]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

    kde_skl = KernelDensity(bandwidth=150)
    kde_skl.fit(xy_train)


    z = np.exp(kde_skl.score_samples(xy_sample))
    zz=np.reshape(z, xx.shape)
    return zz



if __name__ == "__main__":
    filename = '55_31477_47870'

    zz=calAttenMap(filename)
    zz=cv2.resize(zz,(8000,8000))

    np.asve('./attenMap/{}.npy'.format(filename),zz)



