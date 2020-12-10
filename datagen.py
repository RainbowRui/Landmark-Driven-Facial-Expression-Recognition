'''
This file is for data preprocessing
'''
import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils import data
from torchvision import transforms

train_transform=transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

validation_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

# flipped landmarks' corresponding indices
flip_land_ind_ = np.zeros(68, dtype=np.int32)
for m in range(17):
    flip_land_ind_[m] = 16 - m
for m in range(17, 27):
    flip_land_ind_[m] = 43 - m
for m in range(27, 31):
    flip_land_ind_[m] = m
for m in range(31, 36):
    flip_land_ind_[m] = 66 - m
flip_land_ind_[36] = 45
flip_land_ind_[37] = 44
flip_land_ind_[38] = 43
flip_land_ind_[39] = 42
flip_land_ind_[40] = 47
flip_land_ind_[41] = 46
flip_land_ind_[45] = 36
flip_land_ind_[44] = 37
flip_land_ind_[43] = 38
flip_land_ind_[42] = 39
flip_land_ind_[47] = 40
flip_land_ind_[46] = 41
for m in range(48, 55):
    flip_land_ind_[m] = 102 - m
for m in range(55, 60):
    flip_land_ind_[m] = 114 - m
for m in range(60, 65):
    flip_land_ind_[m] = 124 - m
for m in range(65, 68):
    flip_land_ind_[m] = 132 - m

"""def rot_flip_face(image):
    # Random rotate for image (PIL)
    alpha = np.random.uniform(-15, 15)
    image = image.rotate(alpha)

    # Random flip for image (PIL)
    if np.random.uniform(0,1) > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image"""

def rot_flip_face(image, landmark, width):
    '''
        given a face and its 68 landmarks,
        rotate it with 'alpha',
        and return rotated face with new 68 landmarks.
    '''
    # Random rotate for image and corresponding landmarks, np.random.uniform(-15, 15)
    rot_mat = cv2.getRotationMatrix2D(((width-1)/2, (width-1)/2), 5*np.random.randint(-1, 2), 1) # obtain RotationMatrix2D with fixed center
    new_image = cv2.warpAffine(image, rot_mat, (width, width)) # obtain rotated image
    new_landmark = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2], \
        rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark]) # adjust new_landmarks after rotating
    
    # Random flip for image and corresponding landmarks
    if np.random.uniform(0, 1) > 0.5:
        new_image = cv2.flip(new_image, 1)
        flipped_landmark = new_landmark[flip_land_ind_]
        new_landmark[:, 0] = width - 1 - flipped_landmark[:, 0]
        new_landmark[:, 1] = flipped_landmark[:, 1]

    return new_image, new_landmark
    
class TrainSet(data.Dataset):
    '''
        construct trainset, including images and corresponding labels
    '''
    def __init__(self, image_path, label_path, landmark_num, image_width):
        '''
            initialize TrainSet
        '''
        file = open(image_path, 'r')
        image_names  = file.readlines()
        file.close()
        self.image_names = [os.path.join(k.strip('\n')) for k in image_names]
        self.labels = np.loadtxt(label_path, np.long)

        self.transforms = train_transform

        if len(self.image_names) == self.labels.shape[0]:
            self.num_samples = len(self.image_names)

        self.image_width = image_width

        self.landmark_num = landmark_num
        self.landmark_arrays = np.zeros((self.num_samples, self.landmark_num, 2), np.float32)
        self.have_landmark_arrays = np.zeros((self.num_samples, 1), np.long)
        for k in range(self.num_samples):
            image_name = self.image_names[k]
            lands_name = image_name[:-3] + 'txt'
            if os.path.exists(lands_name):
                self.landmark_arrays[k, :, :] = np.loadtxt(lands_name, np.float32).reshape(-1, 2)
                self.have_landmark_arrays[k, :] = 1
        
    def __getitem__(self, index):
        # get, augment image and landmarks
        image_path = self.image_names[index]
        """image = Image.open(image_path)
        image_array = np.array(image).astype(np.uint8)[:,:,None]
        image_array = np.concatenate((image_array, image_array, image_array), axis=2) # channel: 1 -> 3
        image = Image.fromarray(image_array)
        landmark = self.landmark_arrays[index, ...]
        have_landmark = self.have_landmark_arrays[index, :]"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[:,:,None]
        image = np.concatenate((image, image, image), axis=2)
        landmark = self.landmark_arrays[index, ...]
        if image.shape[0] != self.image_width:
            print(image_path, ': size is wrong')
            return
        image, landmark = rot_flip_face(image, landmark, self.image_width)
        have_landmark = self.have_landmark_arrays[index, :]

        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
        
        # get label
        label = self.labels[index]

        return image, label, landmark, have_landmark, index
    
    def __len__(self):
        return self.num_samples

    def cal_mean_landmark(self):
        self.mean_landmark = np.zeros((self.landmark_num, 2), np.float32)
        for k in range(self.num_samples):
            current_landmark = self.landmark_arrays[k, :, :]
            self.mean_landmark += current_landmark
            flipped_landmark = current_landmark[flip_land_ind_]
            current_landmark[:, 0] = self.image_width - 1 - flipped_landmark[:, 0]
            current_landmark[:, 1] = flipped_landmark[:, 1]
            self.mean_landmark += current_landmark
        self.mean_landmark /= (self.have_landmark_arrays.sum() * 2)
        return self.mean_landmark, self.have_landmark_arrays.sum()

class ValidationSet(data.Dataset):
    '''
        construct validation set, including images and corresponding labels
    '''
    def __init__(self, image_path, label_path):
        '''
            initialize ValidationSet
        '''
        file = open(image_path, 'r')
        image_names  = file.readlines()
        file.close()
        self.image_names = [os.path.join(k.strip('\n')) for k in image_names]
        self.labels = np.loadtxt(label_path, np.long)

        self.transforms = validation_transform

        if len(self.image_names) == self.labels.shape[0]:
            self.num_samples = len(self.image_names)

    def __getitem__(self, index):
        # get image
        image_path = self.image_names[index]
        image = Image.open(image_path)
        image_array = np.array(image).astype(np.uint8)[:,:,None]
        image_array = np.concatenate((image_array, image_array, image_array), axis=2) # channel: 1 -> 3
        image = Image.fromarray(image_array)
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
        
        # get label
        label = self.labels[index]

        return image, label
    
    def __len__(self):
        return self.num_samples

class TestSet(data.Dataset):
    '''
        construct test set, including images and corresponding labels
    '''
    def __init__(self, image_path):
        '''
            initialize TestSet
        '''
        file = open(image_path, 'r')
        image_names  = file.readlines()
        file.close()
        self.image_names = [os.path.join(k.strip('\n')) for k in image_names]

        self.transforms = test_transform

        self.num_samples = len(self.image_names)

    def __getitem__(self, index):
        # get image
        image_path = self.image_names[index]
        image = Image.open(image_path)
        image_array = np.array(image).astype(np.uint8)[:,:,None]
        image_array = np.concatenate((image_array, image_array, image_array), axis=2) # channel: 1 -> 3
        image = Image.fromarray(image_array)
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)

        return image

    def __len__(self):
        return self.num_samples