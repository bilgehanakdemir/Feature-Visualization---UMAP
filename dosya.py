import os
import cv2

############# convert grayscale image to RGB #############################
#for i in range(num_class):
#    for j in range(len(image_files[i])):
#        img = cv2.imread(image_files[i][j],cv2.IMREAD_GRAYSCALE)
#        img = np.dstack((img, img, img))    #<class 'numpy.ndarray'>
#        #print(img.shape)
#        #directory = class_names[i]
#        #print("directory: ",directory)
#        #path = os.path.join(data_dir_3D, directory)
#        #if not os.path.exists(path):
#        #mode = 0o666
#          #os.mkdir(path)
#        print("path: ",path1)
#        cv2.imwrite(path1 + '/' + class_names[i] + '_' + '%d.png' % j, img)


############## splits images to train and val #############################



#path_train = '/scratch/project_2006161/cifar-10/data/MednistRGB_split/train/'
#path_test = '/scratch/project_2006161/cifar-10/data/MednistRGB_split/val/'

#for i in os.listdir(path1):
#    #print("i: ",i)
#    img_file = os.path.join(path1, i)
#    img = cv2.imread(img_file,cv2.IMREAD_COLOR)
#    #print("img shape: ",img.shape)
#    img1 = i.split(".")
#    img2 = img1[0].split("_")
#    img_int = int(img2[1])
#    if img_int < 8000:
#        cv2.imwrite(path_train + img2[0] + '_' + '%d.png' % img_int , img)
#    else:
#        cv2.imwrite(path_test + img2[0] + '_' + '%d.png' % img_int , img)
#        

############# convert grayscale image to RGB #############################
#for i in range(num_class):
#    for j in range(len(image_files[i])):
#        img = cv2.imread(image_files[i][j],cv2.IMREAD_GRAYSCALE)
#        img = np.dstack((img, img, img))    #<class 'numpy.ndarray'>
#        #print(img.shape)
#        directory = class_names[i]
#        print("directory: ",directory)
#        path = os.path.join(data_dir_3D, directory)
#        if not os.path.exists(path):
#        #mode = 0o666
#          os.mkdir(path)
#        print("path: ",path)
#        cv2.imwrite(path + '/' + '%d.jpeg' % j, img)

     


############# split class names under the file of train and validation ########################


#path1 = '/scratch/project_2006161/cifar-10/data/MednistRGB_split/train/'
#path2 = '/scratch/project_2006161/cifar-10/data/MednistRGB_split/val/'
#path3 = '/scratch/project_2006161/cifar-10/data/MednistRGB_split2/val/'
#class_names = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
#m = 0
#for i in range(len(class_names)):
#    for j in os.listdir(path2):
#        m+=1
#        #print("i: ",i)
#        img_file = os.path.join(path2, j)
#        img = cv2.imread(img_file,cv2.IMREAD_COLOR)
#        #print("img shape: ",img.shape)
#        img1 = j.split(".")
#        img2 = img1[0].split("_")
#        print("img2: ",img2)
#        img_int = int(img2[1])
#        directory = class_names[i]
#        path = os.path.join(path3, directory)
#        print("directory:",directory, "img2[0]:", img2[0])
#        if img2[0] == directory:
#            if not os.path.exists(path):
#                os.mkdir(path)
#            cv2.imwrite(path + '/' + img2[0] + '_' + '%d.png' % img_int , img)






















