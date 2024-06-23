Configurations: (specified in main function in file ImageMosaics.py)

1. The input images (paths): img1_path, img2_path

2. Getting correspondences manually or automatically: mode, values are {'auto', 'manual'}

3. Choose whether or not use RANSAC 
```
    # without RANSAC (4th parameters should be in [1, 10])
    H, mask = cv2.findHomography(np.asarray(pt1), np.asarray(pt2), 0, 5.0)
    # RANSAC
    H, mask = cv2.findHomography(np.asarray(pt1), np.asarray(pt2), cv2.RANSAC, 5.0)
```
Notice: In question 1. 3), which requires to warp one image into a "frame" region, mode should be set 'manual' and we need to modify the code in function merge_images as follows:
```
    for j in range(img2.shape[0]):
        for i in range(img2.shape[1]):
            merged_image[j-y_zero][i-x_zero] = img2[j][i]
	for j in range(warped_image.shape[0]):
		for i in range(warped_image.shape[1]):
			if np.any(warped_image[j][i]): ## don't write black pixel
				merged_image[j+min_y-y_zero][i+min_x-x_zero] = warped_image[j][i]
```