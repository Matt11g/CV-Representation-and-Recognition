function[colorLabelIm, textureLabelIm] = compareSegmentations(origIm, bank, textons, winSize, numColorRegions, numTextureRegions)
    [h, w, d] = size(origIm);

    colorLabelIm_flat = kmeans(reshape(origIm, h * w, d), numColorRegions);
    colorLabelIm = reshape(colorLabelIm_flat, h, w);

    textonHist = extractTextonHists(rgb2gray(origIm), bank, textons, winSize);
    textureLabelIm_flat = kmeans(reshape(textonHist, h * w, size(textonHist, 3)), numTextureRegions);
    textureLabelIm = reshape(textureLabelIm_flat, h, w);
end