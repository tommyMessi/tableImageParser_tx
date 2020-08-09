
import cv2
import numpy as np
import os
import math

def line_row_gen(img_path):

    img = cv2.imread( img_path )
    img_temp = np.ones_like(img) *255

    gray = cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny( gray,50,150,apertureSize = 3 )
    # cv2.imshow('edge', edges)
    # cv2.imshow('binary',binary)
    # cv2.waitKey(0)

    minLineLength = 100
    maxLineGap = 100
    lines = cv2.HoughLinesP( binary,1,np.pi/180,100,minLineLength=minLineLength,maxLineGap=maxLineGap )
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line( img,( x1,y1 ),( x2,y2 ),( 0,255,0 ),2 )
    except:
        return img_temp, 0
    # points = [(box[0], box[1]), (box[2],box[1]), (box[2], box[3]), (box[0], box[3])]
    # cv2.fillPoly(image,[np.array(points)],(255,0,0))
    # cv2.imwrite( 'E:/image/myhoughlinesp.jpg',img )
    cv2.imshow( '1',img )
    cv2.waitKey(0)
    return img_temp, 1

def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
      included_angle = 360 - included_angle
  return included_angle

def line_col_gen(img_path):
    AB = [0,0,100,0]

    img = cv2.imread( img_path )
    img_temp = np.ones_like(img) *255

    gray = cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny( gray,50,150,apertureSize = 3 )
    # cv2.imshow('edge', edges)
    # cv2.imshow('binary',binary)
    # cv2.waitKey(0)

    minLineLength = 100
    maxLineGap = 100
    lines = cv2.HoughLinesP( binary,1,np.pi/180,100,minLineLength=20,maxLineGap=50 )
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                CD = [x1, y1, x2, y2]
                angle_cross = angle(AB, CD)
                if angle_cross<90+15 and angle_cross>90-15:
                    cv2.line( img,( x1,y1 ),( x2,y2 ),( 255,0,0 ),2 )
    except:
        return img_temp, 0
    # points = [(box[0], box[1]), (box[2],box[1]), (box[2], box[3]), (box[0], box[3])]
    # cv2.fillPoly(image,[np.array(points)],(255,0,0))
    # cv2.imwrite( 'E:/image/myhoughlinesp.jpg',img )
    cv2.imshow( '2',img )
    cv2.waitKey(0)
    return img_temp, 1

def tx_post(row_path, nrow_path, col_path, ncol_path):
    row_image, is_row_exist = line_row_gen(row_path)
    nrow_image, is_nrow_exist = line_row_gen(nrow_path)
    col_image, is_col_exist = line_col_gen(col_path)
    ncol_image, is_ncol_exist = line_col_gen(ncol_path)


if __name__ == '__main__':
    img_root = r'\result\test'
    col_root = r'\result\col'
    row_root = r'\result\row'
    ncol_root = r'\result\ncol'
    nrow_root = r'\result\nrow'


    img_names = os.listdir(col_root)
    for img_name in img_names:
        col_path = os.path.join(col_root, img_name)
        ncol_path = os.path.join(ncol_root, img_name)
        row_path = os.path.join(row_root, img_name)
        nrow_path = os.path.join(nrow_root, img_name)
        # save_path = os.path.join(save_root, img_name)
        tx_post(row_path, nrow_path, col_path, ncol_path)
