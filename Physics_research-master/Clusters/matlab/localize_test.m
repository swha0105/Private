clc;clear; close all;

xray_path = '/storage/cluster/result/xray_pics';



nx = 256 ;

img_3d = zeros(nx,nx,254);
i=0;
%test = imread([xray_path '/' num2str(i) '.png']);


% 
for i=0:255
    
img_3d(:,:,i+1) = rgb2gray(imread([xray_path '/' num2str(i) '.png']));

end
% % 
volumeViewer

img_label=zeros(256,256,256);

for ix=1:256
    for iy=1:256
        for iz=1:256
            if img_3d(ix,iy,iz) == 255
                img_label(ix,iy,iz) = 1;
            end
        end
    end
end
                
    
% 
% 
% for iz=0:253
%     img_3d(:,:,iz+1) = rgb2gray(imread([xray_path '/' num2str(i) '.png']));
%     test = imread([xray_path '/' num2str(i) '.png']);
%     for ix = 1:nx
%     for iy = 1:nx
%         if (test(ix,iy,1)>200 && test(ix,iy,2)^2 + test(ix,iy,3)^2 < 200)
%             test_re(ix,iy,iz+1) = 1;
%         end
%     end
%     end
% end

%imshow(test_re)
            