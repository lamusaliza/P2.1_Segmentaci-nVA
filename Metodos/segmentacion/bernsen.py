import numpy as np

def bernsen(img, cmin=15, n=15, bg="c"):
    dx, dy = img.shape
    imgN = np.copy(img)

    if bg == "claro":
        K = 0
    elif bg == "oscuro":
        K = 255

    w = n // 2

    for i in range(w, dx - w):
        for j in range(w, dy - w):
            block = img[i - w:i + w + 1, j - w:j + w + 1]
            Zlow = np.min(block)
            Zhigh = np.max(block)
            bT = (Zlow + Zhigh) / 2.0
            cl = Zhigh - Zlow

            if cl < cmin:
                wBTH = K
            else:
                wBTH = bT

            if img[i, j] < wBTH:
                imgN[i, j] = 0
            else:
                imgN[i, j] = 255

    return imgN