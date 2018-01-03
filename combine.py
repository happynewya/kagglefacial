import numpy as np
import pandas as pd
FEATURE1=[
'left_eye_center_x'  ,
'left_eye_center_y'  ,
'right_eye_center_x' ,
'right_eye_center_y' ,
'nose_tip_x',
'nose_tip_y',
'mouth_center_bottom_lip_x',
'mouth_center_bottom_lip_y'
]

FEATURE2 = [
'left_eye_inner_corner_x'  ,
'left_eye_inner_corner_y'  ,
'left_eye_outer_corner_x'  ,
'left_eye_outer_corner_y'  ,
'right_eye_inner_corner_x' ,
'right_eye_inner_corner_y' ,
'right_eye_outer_corner_x' ,
'right_eye_outer_corner_y' ,
'left_eyebrow_inner_end_x' ,
'left_eyebrow_inner_end_y' ,
'left_eyebrow_outer_end_x' ,
'left_eyebrow_outer_end_y' ,
'right_eyebrow_inner_end_x',
'right_eyebrow_inner_end_y',
'right_eyebrow_outer_end_x',
'right_eyebrow_outer_end_y',
'mouth_left_corner_x'    ,
'mouth_left_corner_y'    ,
'mouth_right_corner_x'   ,
'mouth_right_corner_y'   ,
'mouth_center_top_lip_x' ,
'mouth_center_top_lip_y' 
]




df1 = pd.read_csv('result1.csv')
df2 = pd.read_csv('result2.csv')
dflook = pd.read_csv('look.csv')
dfsub = pd.read_csv('submission.csv',dtype= {'RowId': np.int32, 'Location':np.float64})

for i in dflook['RowId']:
	imgid = dflook['ImageId'][i-1]
	fname = dflook['FeatureName'][i-1]
	if fname in FEATURE1:
		dfsub.at[i-1, 'Location']= float(df1[fname][int(imgid-1)])
	elif fname in FEATURE2:
		dfsub.at[i-1, 'Location']= float(df2[fname][int(imgid-1)])

dfsub.to_csv('mysubmission.csv', index=False)

