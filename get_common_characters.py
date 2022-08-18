# most common characters 

import os 
import shutil 

top100 = ['的','一','是','不','了','人','我','在','有','他','这','为','之','大','来','以',
'个','中','上','们','到','说','国','和','地','也','子','时','道','出','而','要','于','就','下',
'得','可','你','年','生','自','会','那','后','能','对','着','事','其','里','所','去','行','过',
'家','十','用','发','天','如','然','作','方','成','者','多','日','都','三','小','军','二','无',
'同','么','经','法','当','起','与','好','看','学','进','种','将','还','分','此','心','前','面',
'又','定','见','只','主','没','公','从',',','。','!','?']


test_dir_src = '/Users/maxrivera/Desktop/chinese-character-dataset/CASIA-HWDB_Test/Test'
train_dir_src = '/Users/maxrivera/Desktop/chinese-character-dataset/CASIA-HWDB_Train/Train'
test_dir_dst = '/Users/maxrivera/Desktop/chinese-character-dataset/top100/Test'
train_dir_dst = '/Users/maxrivera/Desktop/chinese-character-dataset/top100/Train'




for character in top100:

	for train_folder in os.listdir(train_dir_src):

		if character == train_folder:

			# copy train_folder
			shutil.copytree(os.path.join(train_dir_src, train_folder), os.path.join(train_dir_dst, train_folder))

			# break from the loop 
			break 

	for test_folder in os.listdir(test_dir_src):

		if character == test_folder:

			# copy test folder 
			shutil.copytree(os.path.join(test_dir_src, test_folder), os.path.join(test_dir_dst,test_folder))

			# break from loop 
			break 








