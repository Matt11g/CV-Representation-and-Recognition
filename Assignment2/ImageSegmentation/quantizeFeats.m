 function[labelIm] = quantizeFeats(featIm, meanFeats)
    %fearIm: h*w*d,  meanFeats: k*d
    %labelIm: h*w, each (1..k)

    %[h, w, d] = size(featIm);
    %[k, d] = size(meanFeats);
    labelIm = zeros(size(featIm, 1), size(featIm, 2));
    
    for h=1:size(featIm, 1)
        dist = dist2(reshape(featIm(h, :, :), size(featIm, 2), size(featIm, 3)), meanFeats);
        [~, labelIm(h, :)] = min(dist, [], 2);
    end

 end