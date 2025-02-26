# prepare training dataset. The images and labels are in numpy array

import os
import numpy as np
import h5py
import copy
import torch
from tqdm import tqdm, trange

class LoadDataSet:
    def __init__(self, data_name, data_path, min_label, max_label, img_size=64, max_num_img_per_label=1e30, num_img_per_label_after_replica=0):
        """
        data_name: the name of the dataset; must be one of ['RC-49', 'Cell200', 'UTKFace', 'SteeringAngle']
        data_path: the path to the h5 file
        min_label: the minimum label
        max_label: the maximum label
        img_size: image size
        max_num_img_per_label: the maximum number of images for each distinct label that will be used for training
        num_img_per_label_after_replica: the number of images for each distinct label that will be used for training
        """
        
        self.data_name = data_name
        self.data_path = data_path
        self.min_label = min_label
        self.max_label = max_label
        self.img_size = img_size
        self.max_num_img_per_label = max_num_img_per_label
        self.num_img_per_label_after_replica = num_img_per_label_after_replica
        
        ## load the entire dataset from h5 file
        with h5py.File(self.data_path+'/{}_{}x{}.h5'.format(self.data_name, self.img_size, self.img_size), 'r') as hf:
            if self.data_name == "RC-49":
                ## load h5 file
                self.labels_all = hf['labels'][:].astype(float)
                self.images_all = hf['images'][:]
                self.indx_train = hf['indx_train'][:]
                hf.close()
                print("\n Loaded entire RC-49 dataset: {}x{}x{}x{}".format(self.images_all.shape[0], self.images_all.shape[1], self.images_all.shape[2], self.images_all.shape[3]))

            elif self.data_name == "UTKFace":
                ## load h5 file
                self.labels_all = hf['labels'][:].astype(float)
                self.images_all = hf['images'][:]
                hf.close()
                print("\n Loaded entire UTKFace dataset: {}x{}x{}x{}".format(self.images_all.shape[0], self.images_all.shape[1], self.images_all.shape[2], self.images_all.shape[3]))
                
            elif self.data_name == "Cell200":
                ## load h5 file
                self.labels_all = hf['CellCounts'][:].astype(float)
                self.images_all = hf['IMGs_grey'][:]
                hf.close()        
                print("\n Loaded entire Cell200 dataset: {}x{}x{}x{}".format(self.images_all.shape[0], self.images_all.shape[1], self.images_all.shape[2], self.images_all.shape[3]))
            
            elif self.data_name == "SteeringAngle":
                ## load h5 file
                self.labels_all = hf['labels'][:].astype(float)
                self.images_all = hf['images'][:]
                hf.close()
                print("\n Loaded entire SteeringAngle dataset: {}x{}x{}x{}".format(self.images_all.shape[0], self.images_all.shape[1], self.images_all.shape[2], self.images_all.shape[3]))
                
            else:
                raise ValueError("Not Supported Dataset!")
        
        if self.data_name == "SteeringAngle":
            indx = np.where((self.labels_all>self.min_label)*(self.labels_all<self.max_label)==True)[0]
            self.min_label_before_shift = np.min(self.labels_all[indx])
            self.max_label_after_shift = np.max(self.labels_all[indx]+np.abs(self.min_label_before_shift))
    
    ## load training data
    def load_train_data(self):
        
        if self.data_name == "RC-49":
            images = self.images_all[self.indx_train]
            labels = self.labels_all[self.indx_train]

            ## Extract a subset from the entire dataset.
            indx = np.where((labels>self.min_label)*(labels<self.max_label)==True)[0]
            labels = labels[indx]
            images = images[indx]
            
            ## for each distinct label, take no more than max_num_img_per_label images
            print("\n The original training set contains {} images with labels in [{},{}]; for each label, select no more than {} images.>>>".format(len(images), self.min_label, self.max_label, self.max_num_img_per_label))
            sel_indx = []
            unique_labels = np.sort(np.array(list(set(labels))))
            for i in range(len(unique_labels)):
                indx_i = np.where(labels == unique_labels[i])[0]
                if len(indx_i)>self.max_num_img_per_label:
                    np.random.shuffle(indx_i)
                    indx_i = indx_i[0:self.max_num_img_per_label]
                sel_indx.append(indx_i)
            sel_indx = np.concatenate(sel_indx)
            
            images = images[sel_indx]
            labels = labels[sel_indx]
            
            print("\r {} images left and there are {} unique labels".format(len(images), len(set(labels))))
            
        elif self.data_name == "UTKFace":
            ## Extract a subset from the entire dataset.
            images = []
            labels = []
            selected_labels = np.arange(self.min_label, self.max_label+1)
            for i in range(len(selected_labels)):
                curr_label = selected_labels[i]
                index_curr_label = np.where(self.labels_all==curr_label)[0]
                images.append(self.images_all[index_curr_label])
                labels.append(self.labels_all[index_curr_label])
            # for i
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels)
            
            ## for each distinct label, take no more than max_num_img_per_label images
            print("\n The original training set contains {} images with labels in [{},{}]; for each label, select no more than {} images.>>>".format(len(images), self.min_label, self.max_label, self.max_num_img_per_label))
            sel_indx = []
            unique_labels = np.sort(np.array(list(set(labels))))
            for i in range(len(unique_labels)):
                indx_i = np.where(labels == unique_labels[i])[0]
                if len(indx_i)>self.max_num_img_per_label:
                    np.random.shuffle(indx_i)
                    indx_i = indx_i[0:self.max_num_img_per_label]
                sel_indx.append(indx_i)
            sel_indx = np.concatenate(sel_indx)
            
            images = images[sel_indx]
            labels = labels[sel_indx]
            
            print("\r {} images left and there are {} unique labels".format(len(images), len(set(labels))))
            
            ## replicate minority samples to alleviate the data imbalance issue
            max_num_img_per_label_after_replica = np.min([self.num_img_per_label_after_replica, self.max_num_img_per_label])
            if max_num_img_per_label_after_replica>1:
                unique_labels_replica = np.sort(np.array(list(set(labels))))
                num_labels_replicated = 0
                print("\n Start replicating minority samples >>>")
                for i in trange(len(unique_labels_replica)):
                    curr_label = unique_labels_replica[i]
                    indx_i = np.where(labels == curr_label)[0]
                    if len(indx_i) < max_num_img_per_label_after_replica:
                        num_img_less = int(max_num_img_per_label_after_replica - len(indx_i))
                        indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
                        if num_labels_replicated == 0:
                            images_replica = images[indx_replica]
                            labels_replica = labels[indx_replica]
                        else:
                            images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                            labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
                        num_labels_replicated+=1
                #end for i
                images = np.concatenate((images, images_replica), axis=0)
                labels = np.concatenate((labels, labels_replica))
                print("\r We replicate {} images and labels.".format(len(images_replica)))
        
        elif self.data_name == "Cell200":
            ## Extract a subset from the entire dataset.
            images = []
            labels = []
            selected_labels = np.arange(self.min_label, self.max_label+1)
            for i in range(len(selected_labels)):
                curr_label = selected_labels[i]
                index_curr_label = np.where(self.labels_all==curr_label)[0]
                images.append(self.images_all[index_curr_label])
                labels.append(self.labels_all[index_curr_label])
            # for i
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels)       
            
            # for each distinct label, take no more than max_num_img_per_label images
            print("\n The original training set contains {} images with labels in [{},{}]; for each label, select no more than {} images.>>>".format(len(images), self.min_label, self.max_label, self.max_num_img_per_label))
            step_size = 2
            selected_labels = np.arange(self.min_label, self.max_label+1, step_size) # only take images with odd labels for training
            n_unique_labels = len(selected_labels)
            images_subset = []
            labels_subset = []
            for i in range(n_unique_labels):
                curr_label = selected_labels[i]
                index_curr_label = np.where(labels==curr_label)[0]
                images_subset.append(images[index_curr_label[0:min(self.max_num_img_per_label, len(index_curr_label))]])
                labels_subset.append(labels[index_curr_label[0:min(self.max_num_img_per_label, len(index_curr_label))]])
            # for i
            images = np.concatenate(images_subset, axis=0)
            labels = np.concatenate(labels_subset)
            
            print("\r {} images left and there are {} unique labels".format(len(images), len(set(labels))))
        
        elif self.data_name == "SteeringAngle":
            indx = np.where((self.labels_all>self.min_label)*(self.labels_all<self.max_label)==True)[0]
            labels = self.labels_all[indx]
            images = self.images_all[indx]
            
            ## replicate minority samples to alleviate the data imbalance issue
            max_num_img_per_label_after_replica = np.min([self.num_img_per_label_after_replica, self.max_num_img_per_label])
            if max_num_img_per_label_after_replica>1:
                unique_labels_replica = np.sort(np.array(list(set(labels))))
                num_labels_replicated = 0
                print("\n Start replicating monority samples >>>")
                for i in trange(len(unique_labels_replica)):
                    curr_label = unique_labels_replica[i]
                    indx_i = np.where(labels == curr_label)[0]
                    if len(indx_i) < max_num_img_per_label_after_replica:
                        num_img_less = int(max_num_img_per_label_after_replica - len(indx_i))
                        indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
                        if num_labels_replicated == 0:
                            images_replica = images[indx_replica]
                            labels_replica = labels[indx_replica]
                        else:
                            images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                            labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
                        num_labels_replicated+=1
                #end for i
                images = np.concatenate((images, images_replica), axis=0)
                labels = np.concatenate((labels, labels_replica))
                print("\r We replicate {} images and labels.".format(len(images_replica)))
            
            print("\r {} images left and there are {} unique labels".format(len(images), len(set(labels))))
        
        else:
            raise ValueError("Not Supported Dataset!")
        
        assert len(labels)==len(images)
        
        print("\n The training set's dimension: {}x{}x{}x{}".format(images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
                
        print("\r Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
        
        labels_norm = self.fn_normalize_labels(labels)
        
        print("\r Range of normalized labels: ({},{})".format(np.min(labels_norm), np.max(labels_norm)))
        
        return images, labels, labels_norm
    
    ## load the evaluation data
    def load_evaluation_data(self):
        if self.data_name == "RC-49":
            images = self.images_all
            labels = self.labels_all
            ## Extract a subset from the entire dataset.
            indx = np.where((labels>self.min_label)*(labels<self.max_label)==True)[0]
            labels = labels[indx]
            images = images[indx]
            
            eval_labels = np.sort(np.array(list(set(labels))))
            
        elif self.data_name in ["UTKFace", "Cell200"]:
            ## Extract a subset from the entire dataset.
            images = []
            labels = []
            selected_labels = np.arange(self.min_label, self.max_label+1)
            for i in range(len(selected_labels)):
                curr_label = selected_labels[i]
                index_curr_label = np.where(self.labels_all==curr_label)[0]
                images.append(self.images_all[index_curr_label])
                labels.append(self.labels_all[index_curr_label])
            # for i
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels)
            
            eval_labels = np.arange(self.min_label, self.max_label+1)
        
        elif self.data_name == "SteeringAngle":
            indx = np.where((self.labels_all>self.min_label)*(self.labels_all<self.max_label)==True)[0]
            labels = self.labels_all[indx]
            images = self.images_all[indx]
             
            num_eval_labels = 2000
            eval_labels = np.linspace(np.min(labels), np.max(labels), num_eval_labels)
            
        else:
            raise ValueError("Not Supported Dataset!")
        
        print("\n The evaluation set's dimension: {}x{}x{}x{}".format(images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
        
        return images, labels, eval_labels
    
            
            
    ## normalize labels into [0,1]
    def fn_normalize_labels(self, input):
        if self.data_name == "SteeringAngle":
            output = input + np.abs(self.min_label_before_shift)
            output = output / self.max_label_after_shift
        else:
            output = input/self.max_label
        assert output.min()>=0 and output.max()<=1.0
        return output
    
    ## de-normalize labels to their original scale
    def fn_denormalize_labels(self, input):
        if self.data_name == "SteeringAngle":
            output = input*self.max_label_after_shift - np.abs(self.min_label_before_shift)
        else:
            if isinstance(input, np.ndarray):
                output = (input*self.max_label).astype(int)
            elif torch.is_tensor(input):
                output = (input*self.max_label).type(torch.int)
            else:
                output = int(input*self.max_label)
        return output



# example
if __name__ == "__main__":
    file_path = 'C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49' 
    handler = LoadDataSet(data_name="RC-49", data_path=file_path, min_label=0, max_label=90, img_size=64, max_num_img_per_label=25, num_img_per_label_after_replica=0)
    
    # 读取并输出数据
    data = handler.load_train_data()
    
    data2 = handler.load_evaluation_data()
    