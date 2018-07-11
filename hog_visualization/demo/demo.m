

%pic_1 = imread('1.jpg');
%imwrite(pic_1, '1.png');


%pic_1 = im2double(pic_1);
pic_1 = im2double(imread('1.jpg'));

size_a = size(pic_1);
size_a = size_a(1:2);
pic_1 = imresize(pic_1, size_a);

feature_pic_1 = features(pic_1, 8);
feature_pic_1 = feature_pic_1(:);
fp_sort_1 = sort(feature_pic_1, 'descend');

%bow_1 = im2double(imread('/Users/macx/Desktop/MY PA/MY 102A/my paper/experi/result/bow_2.png'));
bow_1 = im2double(imread('bow_1.png'));
bow_1 = imresize(bow_1, size_a);
feature_bow_1 = features(bow_1, 8);
feature_bow_1 = feature_bow_1(:);
fb_sort_1 = sort(feature_bow_1, 'descend');

%hog_1 = im2double(imread('/Users/macx/Desktop/MY PA/MY 102A/my paper/experi/result/hog_1.png'));
hog_1 = im2double(imread('hog_1.png'));
hog_1 = imresize(hog_1, size_a);
feature_hog_1 = features(hog_1, 8);
feature_hog_1 = feature_hog_1(:);
fh_sort_1 = sort(feature_hog_1, 'descend');

cnn_1 = im2double(imread('cnn_1_20.png'));
cnn_1 = imresize(cnn_1, size_a);
feature_cnn_1 = features(cnn_1, 8);
%判断是否在小块内
feature_cnn_1 = feature_cnn_1(:);
fc_sort_1 = sort(feature_cnn_1, 'descend');


%top_num = size(fp_sort_1(:));
%fp_sort_1 = fp_sort_1(:);

%fb_sort_1 = fb_sort_1(:);
%fb_sort_1 = fb_sort_1(1:top_num);

%fh_sort_1 = fh_sort_1(:);
%fh_sort_1 = fh_sort_1(1:top_num);

%fc_sort_1 = fc_sort_1(:);
%fc_sort_1 = fc_sort_1(1:top_num);

disp('visualize with hoggle, hog VS bow');
hog_pic = norm(fp_sort_1 - fh_sort_1)/norm(fp_sort_1)
bow_pic = norm(fp_sort_1 - fb_sort_1)/norm(fp_sort_1)
%hold on;
%plot(fp_sort_1, 'k');
%plot(fh_sort_1, 'r');
%plot(fb_sort_1, 'b');
%hold off;

%bow_pic = norm(feature_pic_1(:) - feature_bow_1(:))/norm(feature_pic_1(:))
%hog_pic = norm(feature_pic_1(:) - feature_hog_1(:))/norm(feature_pic_1(:))

cnn_1 = im2double(imread('cnn_1_20.png'));
cnn_1 = imresize(cnn_1, size_a);
feature_cnn_1 = features(cnn_1, 8);
feature_cnn_1 = feature_cnn_1(:);
fc_sort_1 = sort(feature_cnn_1, 'descend');

cnn_hog_1 = im2double(imread('cnn_hog_1_3.png'));
cnn_hog_1 = imresize(cnn_hog_1, size_a);
feature_cnn_hog_1 = features(cnn_hog_1, 8);
feature_cnn_hog_1 = feature_cnn_hog_1(:);
fch_sort_1 = sort(feature_cnn_hog_1, 'descend');

disp('visualize with goggle(cnn), hog VS cnn');
cnn_pic = norm(fp_sort_1 - fc_sort_1)/norm(fp_sort_1)
cnn_hog_pic = norm(fp_sort_1 - fch_sort_1)/norm(fp_sort_1)
hold on;
plot(fp_sort_1, 'k');
plot(fc_sort_1, 'r');
plot(fch_sort_1, 'b');
hold off;

disp('visualize with hog, hoggle VS goggle');
hog_pic = norm(fp_sort_1 - fh_sort_1)/norm(fp_sort_1)
cnn_hog_pic = norm(fp_sort_1 - fch_sort_1)/norm(fp_sort_1)
hoggle_goggle = norm(fh_sort_1 - fch_sort_1)


