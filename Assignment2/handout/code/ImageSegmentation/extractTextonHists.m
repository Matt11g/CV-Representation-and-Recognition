function[featIm] = extractTextonHists(origIm, bank, textons, winSize)
    filteredIm = zeros(size(origIm, 1), size(origIm, 2), size(bank, 3));
    for i = 1:size(bank, 3)
        filteredIm(:, :, i) = imfilter(origIm, bank(:, :, i));
    end

    labelIm = quantizeFeats(filteredIm, textons);

    [h, w] = size(origIm);
    featIm = zeros(h, w, size(textons, 1));
    for i = 1:h
        for j = 1:w
            halfWinSize = fix(winSize / 2);
            window = labelIm(max(i-halfWinSize, 1):min(i+halfWinSize, h), max(j-halfWinSize, 1):min(j+halfWinSize, w));
            freq = tabulate(window(:));
            for k = 1:size(freq, 1)
                featIm(i, j, freq(k, 1)) = freq(k, 2);
            end
        end
    end
end