function bmap = edgeOrientedFilters(im)
    [mag, theta] = orientedFilterMagnitude(im);
    mag2 = mag.^0.7;  %rescale
    bmap = nonmax(mag2, theta);
end