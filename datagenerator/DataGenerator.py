from __future__ import print_function
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
import scipy.ndimage as ndimage
import SimpleITK as sitk
from os.path import join

######################################################################
########## SimpleITK data augmentation helper functions ##############
######################################################################

def resample(image, transform, interpolator = sitk.sitkLinear, default_value = 0.0):
    reference_image = image
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)

def affine_translate_3d(transform, x_translation=0, y_translation=0, z_translation=0):
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetTranslation((x_translation, y_translation, z_translation))
    return new_transform

def affine_scale_3d(transform, x_scale=1.0, y_scale=1.0, z_scale=1.0):
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[0,0] = x_scale
    matrix[1,1] = y_scale
    matrix[2,2] = z_scale
    new_transform.SetMatrix(matrix.ravel())
    return new_transform

def affine_rotate_3d(transform, degrees=0.0, rot_axis = 'x'):
    parameters = np.array(transform.GetParameters())
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((3,3))
    radians = -np.pi * degrees / 180.
    if(rot_axis=="x" or rot_axis=="X"):
        rotation = np.array([[1, 0, 0],[0, np.cos(radians), -np.sin(radians)],[0, np.sin(radians), np.cos(radians)]])
    elif(rot_axis=="y" or rot_axis=="Y"):
        rotation = np.array([[np.cos(radians), 0, -np.sin(radians)],[0, 1, 0],[ np.sin(radians), 0, np.cos(radians)]])
    elif(rot_axis=="z" or rot_axis=="Z"):
        rotation = np.array([[np.cos(radians), -np.sin(radians), 0],[np.sin(radians), np.cos(radians), 0],[0, 0, 1]])
    else:
        print("invalid rotation axis, taking X-axis as default rot_axis")
        rotation = np.array([[1, 0, 0],[0, np.cos(radians), -np.sin(radians)],[0, np.sin(radians), np.cos(radians)]])        
    new_matrix = np.dot(rotation, matrix)
    new_transform.SetMatrix(new_matrix.ravel())
    return new_transform

def affine_shear_3d(transform, sh_x_y=0, sh_y_x=0, sh_x_z=0, sh_z_x=0, sh_y_z=0, sh_z_y=0):
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[0,1] = sh_x_y
    matrix[0,2] = sh_x_z
    matrix[1,0] = sh_y_x
    matrix[1,2] = sh_y_z
    matrix[2,0] = sh_z_x
    matrix[2,1] = sh_z_y
    new_transform.SetMatrix(matrix.ravel())
    return new_transform

def affine_flip_3d(transform, img_center, flip_axis='x'):  
    flipped_transform = sitk.AffineTransform(transform)
    flipped_transform.SetCenter(img_center)
    if(flip_axis=='x' or flip_axis=='X'):
        flipped_transform.SetMatrix([-1,0,0,0,1,0,0,0,1])
    elif(flip_axis=='y' or flip_axis=='Y'):
        flipped_transform.SetMatrix([1,0,0,0,-1,0,0,0,1])
    elif(flip_axis=='z' or flip_axis=='Z'):
        flipped_transform.SetMatrix([1,0,0,0,1,0,0,0,-1])
    else:
        print("Invalid flip axis, taking x-axis as default")
        flipped_transform.SetMatrix([-1,0,0,0,1,0,0,0,1])
    return flipped_transform

####################################################################
######################################################################

#############################################################################################################
############ BraTS Segmentation Data Generator ###########################################################
######################################################################################################


class BraTSSegDataGenerator(Dataset):
    
    
    def __init__(self, list_IDs, dim_x, dim_y, dim_z, main_path, num_classes, modalities, weight_decay, init_epoch, augment, img_cent, trans_range, rot_range, scal_range, shea_range, interpolator, default_val):
        #'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        self.list_IDs = list_IDs
        
        self.main_path = main_path

        self.num_classes = num_classes                

        self.modalities = modalities
        
        self.sample_decay = weight_decay
                
        self.epch = init_epoch

        self.class_weight = self.calculate_weights()
        self.class_weight_decay = self.class_weight

        self.augment = augment
        self.img_cent = img_cent
        self.trans_range = trans_range
        self.rot_range = rot_range
        self.scal_range = scal_range
        self.shea_range = shea_range        
        self.interpolator = interpolator
        self.default_val = default_val


    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, idx):
           
        # Find list of IDs
        ID = self.list_IDs[idx]

        # Generate data
        X, y = self.__read_data(join(self.main_path,ID,ID))

        X = X.astype('float32')                
        y = y.astype('uint8')
                
        #print(X.shape)

        return {'input':X, 'output':y}
            
                    
    def __augment_data(self,my_img):
        
        if np.random.randint(0,2,1) > 0:
            
            
            translate = np.random.uniform(-self.trans_range,self.trans_range,3)
            scale = np.random.uniform(1-self.scal_range,1+self.scal_range,3)
            shear = np.random.uniform(-self.shea_range,self.shea_range,6)
            
            rot_dic = ['x','y','z']
            rot_axis = rot_dic[np.random.randint(0,3)]
            rotate = np.random.uniform(-self.rot_range,self.rot_range)
                       
            flip_dic = ['x','z']
            flip_axis = flip_dic[np.random.randint(0,2)]
                
            affine = sitk.AffineTransform(3)
            
            affine_r  = affine_rotate_3d(affine, rotate, rot_axis)
            affine_t  = affine_translate_3d(affine, translate[0], translate[1], translate[2])
            affine_sc = affine_scale_3d(affine,scale[0],scale[1],scale[2])
            affine_f  = affine_flip_3d(affine,self.img_cent,flip_axis)
            #affine_sh = affine_shear_3d(affine, sh_x_y=shear[0], sh_y_x=shear[1], sh_x_z=shear[2], sh_z_x=shear[3], sh_y_z=shear[4], sh_z_y=shear[5])    
            
            my_transform = []
            
            if np.random.randint(0,2,1) > 0:
                my_transform.append(affine_r)
            if np.random.randint(0,2,1) > 0:
                my_transform.append(affine_t)
            if np.random.randint(0,2,1) > 0:
                my_transform.append(affine_sc)
            if np.random.randint(0,2,1) > 0:
                my_transform.append(affine_f)
            #if np.randon.randint(0,2,1) > 0:
            #    my_transform.append(affine_sh)
                
            if len(my_transform) > 0:
            
                orde = np.arange(len(my_transform))
                np.random.shuffle(orde)
                
                composite = sitk.Transform(3, sitk.sitkComposite)
                
                for i in orde:
                    composite.AddTransform(my_transform[i])

                aug_img = []
    
                for i in range(len(my_img)-1):
                    temp = resample(my_img[i], composite, sitk.sitkLinear, self.default_val)
                    aug_img.append(temp)            
                temp = resample(my_img[-1], composite, sitk.sitkNearestNeighbor, self.default_val)
                aug_img.append(temp)

            else:
                
                aug_img = my_img
        
        else:
            
            aug_img = my_img
            

        return aug_img
    
    
    def __read_data(self,data_path):
   
        img = []
        
        for i,m in enumerate(self.modalities): 
            p = '{}_{}.nii.gz'.format(data_path, m)
            img.append(sitk.ReadImage(p))
            
        if self.augment:
            img = self.__augment_data(img)
 
        X = np.empty((len(self.modalities)-1, self.dim_x, self.dim_y, self.dim_z))
        
        for i in range(len(img)-1):
            temp = sitk.GetArrayFromImage(img[i]).transpose((2,1,0))
            temp = self._normalize(temp)
            X[i,:,:,:] = temp[28:-28, 20:-20, 2:-1]
            
        X = X.astype('float32')

        temp = sitk.GetArrayFromImage(img[-1]).transpose((2,1,0))        
        y = temp[28:-28, 20:-20, 2:-1]

        return X,y

                
    def get_weight(self):

        self.class_weight_decay = (self.class_weight * (self.sample_decay ** self.epch)) + 1
        print("===> Sample Weight for Epoch {}: {}".format(self.epch, self.class_weight_decay.tolist()))
        
        self.epch += 1

        return self.class_weight_decay


    def calculate_weights(self):
        weight = np.zeros(self.num_classes, dtype='int64')
        for ID in self.list_IDs:
            ID_path = join(self.main_path,ID,ID)           	
            p = '{}_{}.nii.gz'.format(ID_path, 'seg')
            img = sitk.ReadImage(p)
            data = sitk.GetArrayFromImage(img)   
            uniq, cnt = np.unique(data, return_counts=True)
            weight[uniq] += cnt
        weight += 1
        weight = np.sum(weight, dtype='int64')*np.reciprocal(weight, dtype='float64')
        return weight


    @staticmethod
    def _normalize(raw_data):
        mask = raw_data > 0
        mu = raw_data[mask].mean()
        sigma = raw_data[mask].std()
        data = (raw_data - mu) / sigma
        #data = np.clip(data,-3,3)
        #data = (255 * ((data + 3) / (6))).astype('uint8')
        data = np.clip(data, np.min(data),3)
        data = (data + (-np.min(data))) / (3-np.min(data))
        #data = data.astype('float32')
        #data = data / 255
        return data