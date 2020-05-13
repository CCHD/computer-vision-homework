import cv2
import numpy as np
from scipy.ndimage import filters
import argparse

class harrisCornerDetector:
    def __init__(self, kernelSize=3, k=0.04, blockSize=2, thredshold=0.01):
        self.kernelSize = kernelSize
        self.k = k
        self.blockSize = blockSize
        self.threshold = thredshold

    def computeIx_Iy(self, img):
        Ix = np.zeros(img.shape)
        Iy = np.zeros(img.shape)
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, self.kernelSize)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, self.kernelSize)

        return (Ix, Iy)

    def compute_harris_response(self, img, path):
        Ix, Iy = self.computeIx_Iy(img)

        Mxx = filters.gaussian_filter(Ix*Ix, self.blockSize)
        Mxy = filters.gaussian_filter(Ix*Iy, self.blockSize)
        Myy = filters.gaussian_filter(Iy*Iy, self.blockSize)

        R = np.zeros(img.shape)
        height, weight = img.shape
        lambda_max = np.zeros(img.shape)
        lambda_min = np.zeros(img.shape)
        for row in range(height):
            for col in range(weight):
                M = np.array([[Mxx[row][col], Mxy[row][col]], [Mxy[row][col], Myy[row][col]]])
                l, _ = np.linalg.eig(M)
                lambda_max[row][col] = l[0] if l[0]>l[1] else l[1]
                lambda_min[row][col] = l[1] if l[0]>l[1] else l[0]

        cv2.namedWindow('lambda_max', cv2.WINDOW_AUTOSIZE)
        egeinMax = lambda_max.max()
        egeinMin = lambda_min.min()
        lambda_max_img = (lambda_max-egeinMin) * 255. / (egeinMax - egeinMin)
        lambda_min_img = (lambda_min - egeinMin) * 255. / (egeinMax - egeinMin)

        lambda_max_img = lambda_max_img.astype(np.uint8)
        lambda_min_img = lambda_min_img.astype(np.uint8)
        cv2.imshow('lambda_max', lambda_max_img)
        cv2.namedWindow('lambda_min', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('lambda_min', lambda_min_img)

        # R = det(M) - k*trace^2(M)
        # det(M) = lambda1 * lambda2, trace(M) = lambda1 + lambda2
        # 韦达定理
        detM = Mxx*Myy-Mxy**2
        traceM = Mxx + Myy
        R = detM - self.k * np.power(traceM, 2)

        R_max = R.max()
        R_min = R.min()
        cv2.namedWindow('R', cv2.WINDOW_AUTOSIZE)
        Rimg = (R-R_min) * 255. / (R_max - R_min)
        Rimg = Rimg.astype(np.uint8)
        R_heatmap = cv2.applyColorMap(Rimg, cv2.COLORMAP_JET)
        cv2.imshow('R', R_heatmap)

        #write to file
        cv2.imwrite(path+'_lambda_max.png', lambda_max_img)
        cv2.imwrite(path+'_lambda_min.png', lambda_min_img)
        cv2.imwrite(path+'_R.png', R_heatmap)
        return R

    def find_harris_point(self, img, path):
        R = self.compute_harris_response(img, path)
        cornerThreshold = self.threshold * R.max()
        height, weight = img.shape
        points = []
        for row in range(height):
            for col in range(weight):
                # the R value of the Point(row, col) is larger than the threshold
                if R[row][col] > cornerThreshold:
                    points.append((row, col))
        return points

    def plot_harris_point(self, path):
        src = cv2.imread(path)
        img = src.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        path_post = path.split('.')[-1]
        path = path[:len(path)-len(path_post)]
        harris_points = self.find_harris_point(gray, path)
        for point in harris_points:
            img[point[0], point[1], :]=[0,0,255]
        cv2.namedWindow('resultImg', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('resultImg', img)
        cv2.imwrite(path + '_result.png', img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

# add parser
parser = argparse.ArgumentParser(description='HarrisCornerDetector')
parser.add_argument('-file_path', type=str, help='待检测的文件路径')

if __name__ == '__main__':
    harrisCornerDetector = harrisCornerDetector()
    args = parser.parse_args()
    harrisCornerDetector.plot_harris_point(args.file_path)