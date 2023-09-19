import cv2
import time
import os
import numpy as np


def creat_template(tmp_img):
    tmp_edge = cv2.Canny(tmp_img, 20, 100)
    tmp_pic = cv2.cvtColor(tmp_edge, cv2.COLOR_GRAY2RGB)
    return tmp_pic

def find_template(tmp_model,src_img):
    src_edge = cv2.Canny(src_img, 20, 100)
    src_pic = cv2.cvtColor(src_edge, cv2.COLOR_GRAY2RGB)
    res = cv2.matchTemplate(src_pic, tmp_model, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc, max_val



src_paths = "../robustness_test/image3/"
tmp_path = "../paper_images_channel3/image3mod.bmp"
save_path = "../robustness_test/"

tmp_img = cv2.imread(tmp_path, 0)
tmp_model = creat_template(tmp_img)
th, tw = tmp_img.shape
times = []

src_images = os.listdir(src_paths)
Note = open(save_path+'S-B_opencv_result.txt', mode='w')
for src_path in src_images:
    src_img = cv2.imread(src_paths+src_path, 0)
    start = time.time()
    max_loc, max_val = find_template(tmp_model, src_img)
    end = time.time()
    total_time = float(end - start)*1000
    times.append(total_time)
    print(src_path, " col：", max_loc[0]+tw/2," row：", max_loc[1]+th/2, "  耗时：", total_time, "ms", " score:", max_val)
    line = src_path + "   " + str(max_loc[1] + th / 2) + "   " + str(max_loc[0] + tw / 2) + "   " + str(max_val) + "   " + str(total_time) + "\n"
    Note.writelines(line)
    # 将匹配结果框起来
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    #cv2.rectangle(src_img, tl, br, (0, 0, 255), 2)
    #cv2.imwrite(save_path+src_path, src_img)
print("时间均值", np.mean(times), "标准差：", np.std(times))
Note.close()


