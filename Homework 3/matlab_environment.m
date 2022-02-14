%----------------------------------
% Sean Carda
% ECE 253 - Image Processing
% 11/something/21
% Homework 3 Environment
%----------------------------------
clear;
clc;

% Load the image.
A = imread('Car.tif');
[r, c] = size(A);

% Zero pad the image.
A_pad = uint8(zeros(512, 512));
A_pad(1:r, 1:c) = A;

% Compute the fourier transform of the given image.
A_fft = fftshift(fft2(A_pad));
A_fft_log = log(abs(A_fft));


% Show the original image.
figure(1);
imshow(A);

% Show the padded image.
figure(2);
imshow(A_pad);

figure(3);
imshow(uint8(255 * A_fft_log));
