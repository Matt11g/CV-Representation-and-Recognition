function [mag, theta] = orientedFilterMagnitude(im)
    sigma = 3;
    hsize = sigma * 6;
    [dGx, dGy] = gradient(fspecial('gaussian',hsize,sigma));
    
    N = 6;
    x_filters = cell(1,N);
    y_filters = cell(1,N);
    for i=1:N
        a = pi*(i-1)/N;
        x_filters{i} = cos(a)*dGx;
        y_filters{i} = sin(a)*dGy;
    end
    
    Gxs = cell(1, N);
    Gys = cell(1, N);
    mags = zeros(size(im, 1), size(im, 2), size(im, 3));
    theta = zeros(size(im, 1), size(im, 2));%no use
    for i=1:N
        Gxs{i} = imfilter(im, x_filters{i}, 'replicate');
        Gys{i} = imfilter(im, y_filters{i}, 'replicate');
        mags = mags + sqrt(Gxs{i}.^2 + Gys{i}.^2);
    end
    mag = sqrt(mags(:,:,1).^2 + mags(:,:,2).^2 + mags(:,:,3).^2);
    
end