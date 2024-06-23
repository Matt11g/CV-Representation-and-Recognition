  function[textons] = createTextons(imStack, bank, k)
    %imStack:a cell array of length n containing a series of n grayscale
    %images
    %bank:a filter bank, m*m*d(d total filters, each of size m*m)
    %textons:k*d matrix in which each row is a texton i.e., one quantized filter bank response

    Textons = [];
    % loop through each image
    for i = 1:size(imStack)
        responses = zeros(size(imStack{i}, 1), size(imStack{i}, 2), size(bank, 3));
        for j = 1:size(bank, 3)
            responses(:, :, j) = imfilter(imStack{i}, bank(:, :, j));
        end

        texton = reshape(responses, size(responses, 1) * size(responses, 2), size(bank, 3));
        sample = randperm(size(texton, 1), fix(size(texton, 1) / 1000));
        Textons = [Textons; texton(sample, :)];
    end

    [~, textons] = kmeans(Textons, k);
  end