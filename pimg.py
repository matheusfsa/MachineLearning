import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
##################################
#
# Nome: Matheus Santos Almeida
# Matricula: 201600092332
# E-mail: matheussa@dcomp.ufs.br
#
##################################

# Funções auxiliares
def v_l(res, n):
    return res if n > 1 else res[0]

def er(img, w, e,func):
    n = nchannels(img)
    a, b = np.array(w.shape)//2
    S = np.array([e*(x, y) for x in range(-b, b + 1) for y in range(-a, a + 1) if w[x + a][y + a] != 0])

    def er_pix(x, y):
        rgb = np.array([f(img, x + s[0], y + s[1]) * w[s[0] + a, s[1] + b] for s in S]).reshape(S.shape[0], n)
        return v_l([func(rgb[:, i]) for i in range(n)], n)
    arr = [[er_pix(x, y) for y in range(size(img)[0])] for x in range(size(img)[1])]
    return np.array(arr, np.uint8)


def f(img, x, y):
    return img[min(max(0, x), img.shape[0]-1), min(max(0, y), img.shape[1]-1)]

# Q.2
def imread(filename):
    img = mpimg.imread(filename)
    return np.array(255*img, dtype=np.uint8) if img.dtype == np.float32 else img

# Q.3
def nchannels(img):
    return 1 if len(img.shape) == 2 else img.shape[2]

# Q.4
def size(img):
    return img.shape[1], img.shape[0]

# Q.5
def rgb2gray(img):
    return np.array(np.dot(img, np.array([0.299, 0.587, 0.114])), dtype=np.uint8) if img.shape[2] == 3 else img


# Q.6
def imreadgray(file):
    img = imread(file)
    return rgb2gray(img) if nchannels(img) == 3 else img

# Q.7
def imshow(img):
    if nchannels(img) == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()

# Q.8
def tresh(img, treshold):
    return (img > treshold)*255

# Q.9
def negative(img):
    return 255 - img

# Q.10
def contrast(img, r, m):
    return np.array([[v_l([min(max(r * (x[i] - m) + m, 0), 255) for i in range(nchannels(img))]) for x in row] for row in img], dtype=np.uint8)

# Q.11
def hist(img):
    return np.array([[(np.sum(img == i)) if nchannels(img) == 1 else (np.sum(img[:,:, j] == i)) for j in range(nchannels(img))]for i in range(256)])


# Q.12
# Q.13
def showhist(hs, bin=1):
    colors, somas = ['red', 'green', 'blue'], [0, 0.25, 0.5]
    res = np.array([[sum(hs[i:i + bin, j]) for i in range(0, 256, bin)] for j in range(hs.shape[1])])
    for i in range(hs.shape[1]):
        plt.bar((np.arange(res.shape[1])) + somas[i], res[0], width=0.25, align='center', color=colors[i])
    plt.show()

# Q.14
def histeq(img):
    hs = hist(img).reshape((1, 256))
    total = size(img)[1] * size(img)[0]
    tr = [sum(hs[0, :i+1])/float(total) for i in range(256)]
    return np.array([[tr[a]*255 for a in row] for row in img], dtype=np.uint8)

# Q.15
def convolve(img, mask):
    return er(img, mask, sum)

# Q.16
def maskblur():
    return 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1,2, 1]])

# Q.17
def blur(img):
    return convolve(img, maskblur())

# Q.18
def seSquare():
    return np.ones([3, 3], np.uint8)

# Q.19
def seCross3():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

# Q.20
def erode(img, eb):
    return er(img, eb, 1, np.amin)

# Q.21
def dilate(img, eb):
    return er(img, eb, -1, np.amax)