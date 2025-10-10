
import os
import cv2
import numpy as np

image1_folder = 'D:\CrowdsourcingGeoLocalization\CrowdsourcingGeoLocalization\Data\ANU_data_small\polarmap_cvhk_all'
image2_folder = 'D:\CrowdsourcingGeoLocalization\CrowdsourcingGeoLocalization\Data\ANU_data_small\streetview_cvhk_all'
save_folder = 'D:\CrowdsourcingGeoLocalization\CrowdsourcingGeoLocalization\Data\ANU_data_small\polarmap_cvhk_all_caijian'


image1_files = [os.path.splitext(f)[0].replace('_satView', '') for f in os.listdir(image1_folder) if f.endswith('_satView.png')]


for file_name in image1_files:

    image1_path = os.path.join(image1_folder, file_name + '_satView.png')
    image2_path = os.path.join(image2_folder, file_name + '_grdView_pano.png')


    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)


    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)


    if len(matches) < 10:
        print(f"Skipping {file_name} due to insufficient matches.")
        continue


    offsets = [kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in matches if m.queryIdx < len(kp1) and m.trainIdx < len(kp2)]
    shift_amount = int(np.mean(offsets))


    exceeding_part = image1[:, -shift_amount:]
    remaining_part = image1[:, :-shift_amount]


    wrapped_pano_image = np.hstack((exceeding_part, remaining_part))


    save_path = os.path.join(save_folder, file_name + '_satView_wrapped.png')

    cv2.imwrite(save_path, wrapped_pano_image)
