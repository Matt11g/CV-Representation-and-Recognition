function bmap = edgeGradient(im)
    sigma = 3;
    [mag, theta] = gradientMagnitude(im, sigma);
    mag2 = mag.^0.7;  %rescale
    bmap = nonmax(mag2, theta);
end