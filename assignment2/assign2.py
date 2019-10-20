################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    y = np.array([0.299, 0.587, 0.114])
    img_gray = img_color @ y
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    n = 5
    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    x = np.arange(-1*n, n+1)

    filt = np.exp((x**2)/-2/ (sigma**2))
    for i in range(len(filt)):
        if np.abs(filt[i]) >= 0.001:
            filt = filt[i:len(filt)-i]
            break
    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    norm = np.ones_like(img)
    norm = convolve1d(norm, filt, 1, np.float64, 'constant', 0, 0)
    img_smoothed = convolve1d(img, filt, 1, np.float64, 'constant', 0, 0)
    # normalization
    img_smoothed = img_smoothed / norm
    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(img, sigma)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(img_smoothed.T, sigma).T
    return img_smoothed

################################################################################
#   utility function to find local maximum / find sub-pixel accuracy
################################################################################
def check_local_maxima(x, y, matrix):
    for i in range(-1, 2): # consider 8-neighbour only
        for j in range(-1, 2):
            if matrix[y+j, x+i] > matrix[y, x]:
                return False
    return True

# find & estimate R in sub-pixel value
def quad_approx(x, y, matrix):
    top = matrix[y-1, x] if y > 0 else 0 # matrix[vertical, horizontal]
    left = matrix[y, x-1] if x > 0 else 0
    right = matrix[y, x+1] if x < matrix.shape[1]-1 else 0
    bottom = matrix[y+1, x] if y < matrix.shape[0]-1 else 0
    a = (left + right - 2*matrix[y,x])/2 
    b = (top + bottom - 2*matrix[y,x])/2
    c = (right - left)/2
    d = (bottom - top)/2
    e = matrix[y, x]
    dx = -c/2/(a+1e-8)
    dy = -d/2/(b+1e-8)
    R = a*dx*dx + b*dy*dy + c*dx + d*dy + e
    return (dx, dy, R)

def finite_diff1d(img, rescale=False):
    diff = np.zeros_like(img, dtype=np.float64)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if x == 0: # forward difference
                diff[y, x] = img[y, x+1] - img[y, x]
            elif x == img.shape[1] - 1: # backward difference
                diff[y, x] = img[y, x] - img[y, x-1]
            else:
                diff[y, x] = (img[y, x+1] - img[y, x-1]) * 0.5
    # min-max scale
    if rescale:
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255
    return diff

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner
    kappa = 0.04
    # TODO: compute Ix & Iy
    Ix = finite_diff1d(img)
    Iy = finite_diff1d(img.T).T

    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # TODO: smooth the squared derivatives
    Ix2 = smooth2D(Ix2, sigma)
    Iy2 = smooth2D(Iy2, sigma)
    IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    R = np.zeros_like(img)
    for x in range(R.shape[1]):
        for y in range(R.shape[0]):
            detA = Ix2[y, x]*Iy2[y, x] - IxIy[y,x]**2
            traceA = Ix2[y,x] + Iy2[y,x]
            R[y, x] = detA - kappa * traceA**2 # R = det(A) - k(trace(A)^2)

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    candidates = [] # stores the sub-pixel accuracy pixel value of candidates
    corners = []
    for y in range(R.shape[0]-1): # vertical
        for x in range(R.shape[1]-1): # horizontal
            if x == 0 or y == 0: # ignore border case in local maximum detection
                continue
            if x % 100 == 0 and y % 100 == 0:
                print("x{} y{}".format(x,y))
            if check_local_maxima(x, y, R):
                dx, dy, r = quad_approx(x, y, R)
                #dx, dy = 0, 0
                candidates.append((x+dx, y+dy, r))

    # TODO: perform thresholding and discard weak corners
    for val in candidates:
        if val[2] >= threshold:
            corners.append(val)
    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
