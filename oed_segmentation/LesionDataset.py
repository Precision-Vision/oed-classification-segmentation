import os
import numpy as np
import torch
from torchvision import transforms as T2 
import albumentations as A
from PIL import Image
from xml_to_mask import *

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms: bool = True, one_class_mode=False, linecolors=['65280', '10485760'], batch_no=1, downsample=1, minimum_size=[512, 512], excluded_classes=[]):
        self.root = root
        self.batch_no = batch_no
        self.linecolors = linecolors
        self.excluded_classes = excluded_classes
        paths = self.get_dataset_paths(root, batch_no)
        self.transforms = transforms 
        self.imgs = paths[0]
        self.annotations = paths[1]
        self.downsample = downsample
        self.minimum_size = minimum_size
        self.one_class_mode = one_class_mode

    @property
    def validation_ids(self):
        validation_ids = ['13.2766', '13.2875', '13.2752', '13.2868', '13.2744', '13.2777']
        return validation_ids

    @property
    def binary_classes(self):
        if self.batch_no == 1:
            classes = ['suspicious', 'normal']
            return classes
        if self.batch_no == 2:
            all_classes = ['Cancer', 'Dysplastic', 'Non-dysplastic']
            classes = [c for c in all_classes if c not in self.excluded_classes]
            return classes
        else:
            raise ValueError('Invalid batch number: {0}'.format(self.batch_no))
    
    @property
    def binary_classes_indicies(self):
        indicies = [x + 1 for x in range(len(self.binary_classes))]
        return indicies
    
    @property
    def class_colors(self):
        if self.batch_no == 2:
            colors = {
                'Non-dysplastic': [[0.0, 1.0, 0.0], 'Greens', 'Non-dysplastic'],
                'Dysplastic': [[1.0, 0.5, 0.0], 'rainbow', 'Dysplastic'],
                'Cancer': [[1.0, 0.0, 0.0], 'jet', 'Cancer'],
            }
            return colors
        else:
            raise ValueError('Invalid batch number: {0}'.format(self.batch_no))

    def random_validation_ids(self, ids, n=6):
        validation_ids = []
        ids_range = range(len(ids))
        # print('ids_range: {0}'.format(ids_range))
        rand_idxs = np.random.choice(ids_range, size=n, replace=False)
        for i in rand_idxs:
            # print('rand_idx: {0}'.format(i))
            validation_ids.append(ids[i])
        return validation_ids
    
    @property
    def grading_classes(self):
        if self.batch_no == 1:
            classes = [c for c in ['mild', 'moderate', 'severe'] if c not in self.excluded_classes]
            return classes
        elif self.batch_no == 2:
            # Exclude classes using self.excluded_classes
            classes = [c for c in ['Cancer', 'Dysplastic', 'Non-dysplastic'] if c not in self.excluded_classes]
            return classes
        else:
            raise ValueError('Invalid batch number: {0}'.format(self.batch_no))

    def compose_transforms_a(self, img, target):
        if target['masks'].shape[0] == 0:
            print('No masks found')
            masks = np.zeros((img.size[1], img.size[0], 1))
        else:
            masks = np.array(target['masks']).transpose(1, 2, 0)
        
        if self.transforms is True:
            transform = A.Compose([
                # A.RandomCrop(width=img.size[0], height=img.size[1], p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=0, p=0.5),
                # A.RandomRotate90(p=0.2),
                A.Rotate(limit=180, p=0.5, border_mode=0),
                # A.RandomScale(scale_limit=(0.1, 0.8), p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            transformed = transform(image=np.array(img), mask=masks, bboxes=target['boxes'], labels=target['labels'])
            img = transformed['image']
            target['masks'] = transformed['mask']
            target['boxes'] = transformed['bboxes']
            target['masks'] = target['masks'].transpose(2, 0, 1)
            target['labels'] = transformed['labels']
        # Convert to Tensor
        img = T2.ToTensor()(img)
        
        # img = img.permute(1, 2, 0)
        return img, target

    def get_all_id_pairs(self, root):
        if self.batch_no == 1:
            return self.get_all_id_pairs_batch_01(root)
        elif self.batch_no == 2:
            return self.get_all_id_pairs_batch_02(root)
        else:
            raise ValueError('Invalid batch number: {0}'.format(self.batch_no))

    def get_all_id_pairs_batch_01(self, root):
        all_id_pairs = []
        # Find all images and annotations
        for grading_class in self.grading_classes:
            for _, ids, _ in os.walk(root + '/' + grading_class, topdown=False):
                for id in ids:
                    all_id_pairs.append((grading_class, id))
        return all_id_pairs

    def get_all_id_pairs_batch_02(self, root):
        all_id_pairs = []
        # Find all images and annotations
        for grading_class in self.grading_classes:
            for _, _, ids in os.walk(root + '/' + grading_class, topdown=False):
                # Remove extension from ids
                ids = [x.split('.')[0] for x in ids]
                for id in ids:
                    if len([x for x in ids if x == id]) != 2:
                        print('Id {0} does not have two files (image or annotation)'.format(id))
                        # Remove id from ids
                        ids = np.delete(ids, np.where(ids == id))
                ids = sorted(np.unique(ids))
                for id in ids:
                    all_id_pairs.append((grading_class, id))
        return all_id_pairs

    def get_dataset_paths(self, root, batch_no=1):
        if batch_no == 1:
            return self.get_dataset_paths_batch_01(root)
        elif batch_no == 2:
            return self.get_dataset_paths_batch_02(root)
        else:
            raise ValueError('Invalid batch number: {0}'.format(self.batch_no))

    def get_dataset_paths_batch_01(self, root):
        all_id_pairs = self.get_all_id_pairs(root)
        selected_id_pairs = all_id_pairs
        
        print('all_id_pairs: {0}'.format(all_id_pairs), len(all_id_pairs))
        print('selected_id_pairs: {0}'.format(selected_id_pairs), len(selected_id_pairs))

        # Create image and annotation paths
        image_paths = []
        annot_paths = []
        for grading_class, id in selected_id_pairs:
            image_path = self.root + '{0}/{1}/1.jpg'.format(grading_class, id)
            annot_path = self.root + '{0}/{1}/1.xml'.format(grading_class, id)
            image_paths.append(image_path)
            annot_paths.append(annot_path)
        return image_paths, annot_paths    
    
    def get_dataset_paths_batch_02(self, root):
        all_id_pairs = self.get_all_id_pairs(root)

        training_ids = [x for x in all_id_pairs]
        
        # print('all_id_pairs: {0}'.format(all_id_pairs), len(all_id_pairs))
        # print('training_ids: {0}'.format(training_ids), len(training_ids))

        # Create image and annotation paths
        image_paths = []
        annot_paths = []

        for grading_class, id in training_ids:
            image_path = self.root + '{0}/{1}.jpg'.format(grading_class, id)
            annot_path = self.root + '{0}/{1}.xml'.format(grading_class, id)
            # Check if image_path and annot_path exists
            if os.path.exists(image_path) and os.path.exists(annot_path):
                image_paths.append(image_path)
                annot_paths.append(annot_path)
            else:
                print('Image or annotation path does not exist: {0}, {1}'.format(image_path, annot_path))
        return image_paths, annot_paths

    def __getitem__(self, idx):
        class_ids = []
        img_path = self.imgs[idx]
        annot_path = self.annotations[idx]
        grading_class = self.grading_classes.index(img_path.split('/')[-2])
        
        img = Image.open(img_path).convert("RGB")
        # Scale image down by downsample factor
        # If image is too small (height or width < 512) don't downsample
        downsample = self.downsample
        if img.size[0] > self.minimum_size[0] and img.size[1] > self.minimum_size[1]:
            img = img.resize((int(img.size[0]/downsample), int(img.size[1]/downsample)))
        else:
            downsample = 1

        height, width = img.size 

        masks = xml_to_masks(annot_path, location=(0,0), size=(height, width), downsample=downsample, verbose=False, linecolors=self.linecolors)
        
        bounds = []
        finalmasks = []
        for submask in masks:
            # Get bbox using skimage/Torch
            id = np.unique(submask)[1] 
            if self.one_class_mode and id == 2:
                continue
            submask = submask == id
            pos = np.where(submask) 
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            finalmasks.append(submask)
            bounds.append([xmin, ymin, xmax, ymax]) # pascal_voc format
            class_ids.append(id)
        # TODO: change to tensor = torch.from_numpy(finalmasks) (check dtype of finalmasks)
        masks = torch.as_tensor(finalmasks, dtype=torch.uint8)
        class_ids = np.array(class_ids, dtype=np.int32)
        num_objs = len(class_ids)
        boxes = torch.as_tensor(bounds, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        image_id = torch.tensor([idx])
        if boxes.shape[0] == 0:
            print('No boxes found for image: {0}'.format(img_path))
            area = 0
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['grading_class'] = grading_class

        img, target = self.compose_transforms_a(img, target)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.int64)
        target['grading_class'] = torch.as_tensor(target['grading_class'], dtype=torch.int64)

        return img, target
    
    def reload(self):
        paths = self.get_dataset_paths(self.root, val=self.val, random_val=self.random_val)
        self.imgs = paths[0]
        self.annotations = paths[1]

    def __len__(self):
        return len(self.imgs)