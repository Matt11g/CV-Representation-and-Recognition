function [mag, theta] = gradientMagnitude(im, sigma)
    % This function should take an RGB image as input, smooth the image with Gaussian std=sigma, compute
    % the x and y gradient values of the smoothed image, and output image maps of the gradient magnitude
    % and orientation at each pixel. 

    filteredIm = imgaussfilt(im, sigma);
    
    [Gx, Gy] = gradient(filteredIm);
    
    mags = sqrt(Gx.^2 + Gy.^2);
    mag = sqrt(mags(:,:,1).^2 + mags(:,:,2).^2 + mags(:,:,3).^2);
    
    sintheta = Gx ./ mags;
    costheta = Gy ./ mags;
    thetas = atan(sintheta ./ costheta);
    thetas = thetas  + (pi .* sign(sintheta)) .* sign(-costheta);
    
    [~, max_idx] = max(mags, [], 3);
    theta = zeros(size(thetas, 1), size(thetas, 2));

    for i = 1:3
        theta = theta + thetas(:, :, i) .* (max_idx == i);
    end
end