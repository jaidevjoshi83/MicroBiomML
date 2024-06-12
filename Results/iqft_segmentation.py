#!/usr/bin/env python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def im2arr(file_path):
    # load grayscale image
    img = Image.open(file_path).convert("L") #convert to grayscale; use (LA) for RGB
    return np.array(img)
    
def img2ang(image):
    return image / 255.

def ang2basis(angles, nqubits = 3, theta=8*np.pi/4, transform=1):
    N = 2 ** nqubits
    w = np.exp(1j * 2 * np.pi / N)
    _wp = lambda x: w ** (x)
    _fgray = lambda g : np.array([1, np.exp(g*1j)])
    _wgray = np.array([[1, 1], [1, -1]])
    _frgb = lambda g,b,a : np.array([1, np.exp(g*1j), np.exp(b*1j), np.exp(1j*(b+g)), np.exp(1j*a), np.exp(1j*(a+g)), np.exp(1j*(a+b)), np.exp(1j*(a+b+g))])
    _wrgb = np.array([[         1,    1,       1,        1,      1,       1,       1,       1],
                               [1, _wp(-1), _wp(-2), _wp(-3), _wp(-4), _wp(-5), _wp(-6), _wp(-7)],
                               [1, _wp(-2), _wp(-4), _wp(-6),    1,    _wp(-2), _wp(-4), _wp(-6)],
                               [1, _wp(-3), _wp(-6), _wp(-1), _wp(-4), _wp(-7), _wp(-2), _wp(-5)],
                               [1, _wp(-4),    1,    _wp(-4),    1,    _wp(-4),     1  , _wp(-4)],
                               [1, _wp(-5), _wp(-2), _wp(-7), _wp(-4), _wp(-1), _wp(-6), _wp(-3)],
                               [1, _wp(-6), _wp(-4), _wp(-2),    1,    _wp(-6), _wp(-4), _wp(-2)],
                               [1, _wp(-7), _wp(-6), _wp(-5), _wp(-4), _wp(-3), _wp(-2), _wp(-1)],
    ])

    if (angles.ndim == 2) and transform:
        #angles = np.stack([angles,angles/255,angles/(255*255)],axis=2)
        angles = np.stack([angles,angles,angles],axis=2)

    if angles.ndim == 2:
        _f = _fgray
        W = _wgray
        N = 2 ** 1
    elif angles.ndim == 3:
        if angles.shape[2] == 1:
            angles = np.squeeze(angles)
            _f = _fgray
            W = _wgray
        elif angles.shape[2] == 3:
            _f = _frgb
            W = _wrgb

    nrow, ncol = angles.shape[:2]

    mask = np.zeros((nrow, ncol), dtype='ubyte')

    for i in range(nrow):
        for j in range(ncol):
            if angles.ndim < 3:
                Fm = _f(angles[i,j,] * theta)     # vector
            else:
                Fm = _f(*angles[i,j,] * theta)     # vector
            Sm = np.power( np.abs( np.dot(W, Fm) / N ), 2 )   # vector
            lm = np.argmax(Sm) + 1
            mask[i, j] = lm
    return mask

    #if __name__ == "__main__":
lena_png = 'C:/Users/KOOSK/OneDrive - Cleveland Clinic/IQFT Segmentation/Lena64.png'
lena_py = im2arr(lena_png)
#lena_py = np.flipud(lena_py)

lena_ang = img2ang(lena_py)
lena_seg = ang2basis(lena_ang, 3)

fig, ax = plt.subplots(1,2)
ax[0].imshow(lena_ang, cmap='gray', label='orig')
ax[0].title.set_text("orig")
ax[1].imshow(lena_seg, cmap='jet', label='iqft seg')
ax[1].title.set_text(f"{np.unique(lena_seg)}")
plt.pause(0.1)
plt.show()




    
