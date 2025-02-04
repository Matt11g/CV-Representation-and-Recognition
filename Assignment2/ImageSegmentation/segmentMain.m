%gumballs.jpg, snake.jpg, and twins.jpg, + coins.jpg
gumballs = im2double(imread('gumballs.jpg'));
snake = im2double(imread('snake.jpg'));
twins = im2double(imread('twins.jpg'));
coins = im2double(imread('coins.jpg'));

bank = cell2mat(struct2cell(load('filterBank.mat'))); %Reference: https://www.jianshu.com/p/12ce9032d904
%bank = bank(:,:,1:6:end);
%displayFilterBank(bank);

imStack = {rgb2gray(gumballs), rgb2gray(snake), rgb2gray(twins), rgb2gray(coins)};
k_textons = 17;
textons = createTextons(imStack, bank, k_textons);%%TODO: bank

origIm = {gumballs, snake, twins, coins};
winSize = [12, 6, 9, 10];
numColorRegions = [7, 4, 7, 5];
numTextureRegions = [3, 5, 7, 5];

% [colorLabelIm, textureLabelIm] = compareSegmentations(twins, bank, textons, 7, 7, 7);
% subplot(2, 2, 1);
% imshow(label2rgb(colorLabelIm));
% subplot(2, 2, 2);
% imshow(label2rgb(textureLabelIm));
% [colorLabelIm, textureLabelIm] = compareSegmentations(twins, bank, textons, 22, 7, 7);
% subplot(2, 2, 3);
% imshow(label2rgb(colorLabelIm));
% subplot(2, 2, 4);
% imshow(label2rgb(textureLabelIm));

for i=1:4
   [colorLabelIm, textureLabelIm] = compareSegmentations(origIm{i}, bank, textons, winSize(i), numColorRegions(i), numTextureRegions(i));
   subplot(4, 3, (i-1)*3+1);
   imshow(origIm{i});
   subplot(4, 3, (i-1)*3+2);
   imshow(label2rgb(colorLabelIm));
   subplot(4, 3, (i-1)*3+3);
   imshow(label2rgb(textureLabelIm));
end

