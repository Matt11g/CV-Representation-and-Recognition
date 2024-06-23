import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import map_coordinates
#from vlfeat import sift

def get_correspondences(img1, img2, num_points=4):
    '''
    Input: 
        img, img2: image 
        num_points: number of points we select
    Return:
        points1, points2: the coordinates of selected points
    '''
    print("Select %d points for each image by clicking them" % num_points)
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)
    fig1.imshow(img1)
    fig2.imshow(img2)

    points = plt.ginput(num_points * 2)
    points1 = points[0::2]
    points2 = points[1::2]

    return points1, points2

def compute_homography_paras(pt1, pt2):
    '''
    a function that takes a set of corresponding image points
    and computes the associated 3 * 3 homography matrix H.
    Input: 
        pt1, pt2: coordinates of points
    Return:
        the homography matrix H, of shape (3, 3), where we assume h33 = 1
    '''
    N = len(pt1)
    A = np.zeros((2*N, 8))
    for i in range(N):
        x, y = pt1[i]
        xx, yy = pt2[i]
        A[2*i] = [x, y, 1, 0, 0, 0, -x*xx, -y*xx]
        A[2*i+1] = [0, 0, 0, x, y, 1, -x*yy, -y*yy]
    b = np.reshape(pt2, (-1, ))
    H = np.append(np.linalg.lstsq(A, b, rcond=None)[0], 1).reshape(3, 3)
    return H

def verify_H(img1, img2, H):
    '''
    Verify that the homography matrix your function computes
    is correct by mapping the clicked image points from 
    one view to the other, and displaying them on top 
    of each respective image
    '''
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)
    fig1.imshow(img1)
    fig2.imshow(img2)
    print("Click any pixel in image1 and check the corresponding point in image2")
    
    i = 0
    while i < 5:
        ip = plt.ginput(1)
        pts1 = np.append(np.reshape(ip, (-1, )), 1).reshape(3, 1)
        pts2 = np.dot(H, pts1).reshape(-1, )
        fig1.scatter(ip[0][0], ip[0][1])
        fig2.scatter(pts2[0] / pts2[2], pts2[1] / pts2[2])
        i+=1
    
def warp_between_planes(img1_path, H):
    '''
    Take the recovered homography matrix and an image, and
        return a new image that is the warp of the input image using H
    For color images, warp each RGB channel separately and
        then stack together to form the output.
    To avoid holes in the output, use an inverse warp. 
        1. Warp the points from the source image into the
        reference frame of the destination
        2. compute the bounding box in that new reference fram
        3. sample all points in that destination bounding box
          from the proper coordinates in the source image.
    '''
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    h, w = img1.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=1)
    # Apply the homography transformation to the grid
    new_coords = (H @ coords.T).T # shape: (.., 3)
    # Normalize the transformed coordinates
    new_coords[:, 0] /= new_coords[:, 2]
    new_coords[:, 1] /= new_coords[:, 2]
    new_coords = new_coords[:, :2].reshape(h, w, 2)
    # Compute the bounding box of the warped image
    min_x = np.floor(np.min(new_coords[:, :, 0])).astype(int)
    min_y = np.floor(np.min(new_coords[:, :, 1])).astype(int)
    max_x = np.ceil(np.max(new_coords[:, :, 0])).astype(int)
    max_y = np.ceil(np.max(new_coords[:, :, 1])).astype(int)
    # Interpolate the pixel values in the warped image
    warped_image = np.zeros((int(max_y-min_y+1),int(max_x-min_x+1), 3), dtype=np.uint8)
    for j, row in enumerate(new_coords):
        for i, coord in enumerate(row):
            warped_image[int(coord[1]-min_y), int(coord[0]-min_x)] = img1[j][i]
    # use an inverse warp to avoid holes in the output
    xx, yy = np.meshgrid(np.arange(min_x, max_x+1, 1), np.arange(min_y, max_y+1, 1))
    coords2 = np.stack([xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())], axis=1)
    new_coords2 = (np.linalg.inv(H) @ coords2.T).T
    new_coords2[:, 0] /= new_coords2[:, 2]
    new_coords2[:, 1] /= new_coords2[:, 2]
    new_coords2 = new_coords2[:, :2].reshape(warped_image.shape[0], warped_image.shape[1], 2)
    for j, row in enumerate(new_coords2):
        for i, coord2 in enumerate(row):
            if not(np.any(warped_image[j][i])) and coord2[0] >= 0 and coord2[0] < w and coord2[1] >= 0 and coord2[1] < h:
                warped_image[j][i] = img1[int(coord2[1])][int(coord2[0])]
    return warped_image, (min_x, min_y, max_x, max_y)

def merge_images(warped_image, boundingbox, img2_path):
    '''
    create a merged image showing the mosaic
    '''
    (min_x, min_y, max_x, max_y) = boundingbox
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    hh, ww = img2.shape[:2]
    merged_image = np.zeros((int(max(max_y,hh)-min(min_y,0)+1), (int(max(max_x,ww)-min(min_x,0)+1)), 3), dtype=np.uint8)
    y_zero = min(min_y,0)
    x_zero = min(min_x,0)
    
    for j in range(warped_image.shape[0]):
        for i in range(warped_image.shape[1]):
            if np.any(warped_image[j][i]):
                merged_image[j+min_y-y_zero][i+min_x-x_zero] = warped_image[j][i]
    for j in range(img2.shape[0]):
        for i in range(img2.shape[1]):
            merged_image[j-y_zero][i-x_zero] = img2[j][i]
    return merged_image

def get_correspondences_sift(image1, image2):
    """
    Identify corresponding points between two input images using VLFeat SIFT.
    
    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
    
    Returns:
        list: A list of corresponding point pairs between the two images.
    """
    # Convert images to grayscale
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Compute SIFT keypoints and descriptors for both images
    sift = cv2.SIFT_create(400)
    kp1, desc1 = sift.detectAndCompute(gray1,None)
    kp2, desc2 = sift.detectAndCompute(gray2,None)
    
    # Find matching descriptors using brute-force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test to filter good matches
    good_matches = []
    correspondences1 = []   
    correspondences2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])
            # Identify corresponding points
            correspondences1.append(kp1[m.queryIdx].pt)
            correspondences2.append(kp2[m.trainIdx].pt)
    img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, flags=2, outImg=None)
    plt.imshow(img3)
    plt.show()
    return correspondences1, correspondences2

if __name__ == '__main__':
    # original images
    img1_path = 'uttower2.jpg'
    img2_path = 'uttower1.jpg'
    '''
    # for another images I took
    img1_path = 'img1.jpg'
    img2_path = 'img2.jpg'
    '''
    '''
    # for frame
    img1_path = 'frame_img.jpg'
    img2_path = 'frame.jpg'
    '''
    img1 = plt.imread(img1_path)
    img2 = plt.imread(img2_path)
    mode = 'auto' ## {'auto', 'manual'}
    if mode == 'manual':
        pt1, pt2 = get_correspondences(img1, img2, 4)
        H = compute_homography_paras(pt1, pt2)
    else:
        pt1, pt2 = get_correspondences_sift(img1_path, img2_path)
        # ordinary method RANSAC (4th parameters should be in [1, 10])
        ## H, mask = cv2.findHomography(np.asarray(pt1), np.asarray(pt2), 0, 5.0)
        # RANSAC
        H, mask = cv2.findHomography(np.asarray(pt1), np.asarray(pt2), cv2.RANSAC, 5.0)
    #verify_H(img1, img2, H)
    warped_image, boundingbox = warp_between_planes(img1_path, H)
    merged_image = merge_images(warped_image, boundingbox, img2_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(merged_image)
    plt.show()
