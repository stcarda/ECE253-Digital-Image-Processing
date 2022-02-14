%-------------------------------------
% Sean Carda
% ECE 253 - Image Processing
% Exam Practice
% 11/23/21
%-------------------------------------
clear;
clc;

file_names = ["lena512.tif"; "tolkien.jpg"; "Street,png"; "rohan.jpg"; "gondor.jpg"; "Car.tif"];
for i = 1:length(file_names)
    A = imread(file_names(i));
    [r, c, d] = size(A);
    if d > 1
        A = rgb2gray(A);
    end
    A_hist = imhist(A);

    A_he = histeq(A);
    A_he_hist = imhist(A_he);

    figure(1);
    subplot(2, 3, i);
    plot(A_hist);
    title('Hist for ' + file_names(i));
    
    figure(2);
    subplot(2, 3, i);
    plot(A_he_hist);
    title('EQ Hist for ' + file_names(i));
    
end
