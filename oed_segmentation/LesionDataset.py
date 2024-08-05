import bisect
import json
import os
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T2 
import albumentations as A
from PIL import Image
from xml_to_mask import *
import pycocotools.mask as maskUtils
import coco_utils
from skimage import measure
from utils import path_exists_ext_case_insensitive



class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transforms: bool = True, linecolors=['65280', '10485760'], downsample=1, minimum_size=[512, 512], excluded_classes=[], ignore_non_lesion=True, img_ext='.jpg', annot_ext='.xml', to_tensor=True):
        self.root = root
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.linecolors = linecolors
        self.excluded_classes = excluded_classes
        paths = self.get_dataset_paths(root)
        self.transforms = transforms 
        self.imgs = paths[0]
        self.annotations = paths[1]
        self.downsample = downsample
        self.minimum_size = minimum_size
        self.ignore_non_lesion = ignore_non_lesion
        self.to_tensor = to_tensor

    @staticmethod
    def class_colors():
        colors = {
            'Non-dysplastic': [[0.0, 1.0, 0.0], 'Greens', 'Non-dysplastic'],
            'Dysplastic': [[1.0, 0.5, 0.0], 'rainbow', 'Dysplastic'],
            'Cancer': [[1.0, 0.0, 0.0], 'jet', 'Cancer'],
        }
        return colors

    @staticmethod
    def grading_classes(excluded_classes=[]):
        classes = [c for c in ['Non-dysplastic', 'Dysplastic', 'Cancer'] if c not in excluded_classes]
        return classes

    @staticmethod
    def grading_classes_indices():
        indicies = [x + 1 for x in range(len(LesionDataset.grading_classes()))]
        return indicies
    
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
        all_id_pairs = []
        # Find all images and annotations
        for grading_class in LesionDataset.grading_classes(self.excluded_classes):
            path = root / grading_class
            for _, _, _filenames in os.walk(path, topdown=False):
                # create list of xml files
                xml_files = [f for f in _filenames if f.endswith(self.annot_ext)]
                xml_files_no_ext = [os.path.splitext(x)[0] for x in xml_files]
                # create list of of the remaining files (not xml)
                img_files = [f for f in _filenames if not f.endswith(self.annot_ext)]
                img_files_no_ext = [os.path.splitext(x)[0] for x in img_files]
                # match xml files with corresponding image files
                union = set(xml_files_no_ext) & set(img_files_no_ext)
                for filename in union:
                    xml_file = filename + "." + self.annot_ext
                    # find the corresponding image file and extension from img_files
                    # we do this because the image file extension may vary (e.g. .jpg, .png, .jpeg)
                    try:
                        image_file = [f for f in img_files if os.path.splitext(f)[0] == filename][0]
                    except IndexError:
                        print('Image file not found for annotation file: {0}'.format(xml_file))
                        continue
                    all_id_pairs.append((grading_class, image_file, xml_file))
        return all_id_pairs
        
    def get_dataset_paths(self, root):
        all_id_pairs = self.get_all_id_pairs(root)

        training_ids = [x for x in all_id_pairs]

        # Create image and annotation paths
        image_paths = []
        annot_paths = []

        for grading_class, image_filename, annot_filename in training_ids:
            image_path = self.root / '{0}/{1}'.format(grading_class, image_filename)
            annot_path = self.root / '{0}/{1}'.format(grading_class, annot_filename)
            if image_path.exists() and annot_path.exists():
                image_paths.append(image_path)
                annot_paths.append(annot_path)
            else:
                print('Image or annotation path does not exist: {0}, {1}'.format(image_path, annot_path))
        return image_paths, annot_paths

    def __getitem__(self, idx):
        class_ids = []
        img_path = self.imgs[idx].as_posix()
        annot_path = self.annotations[idx].as_posix()
        grading_class = LesionDataset.grading_classes(self.excluded_classes).index(img_path.split('/')[-2])
        
        img = Image.open(img_path).convert("RGB")
        # Scale image down by downsample factor
        # If image is too small (height or width < 512) don't downsample
        downsample = self.downsample
        if img.size[0] > self.minimum_size[0] and img.size[1] > self.minimum_size[1] and downsample > 1:
            img = img.resize((int(img.size[0]/downsample), int(img.size[1]/downsample)))
        else:
            downsample = 1

        height, width = img.size 

        masks = xml_to_masks(annot_path, location=(0,0), size=(height, width), downsample=downsample, verbose=False, linecolors=self.linecolors)
        
        bounds = []
        finalmasks = []
        for submask in masks:
            id = np.unique(submask)[1] 
            submask = submask == id
            if self.ignore_non_lesion and id == 1:
                continue
            if id == 2:
                id = grading_class + 1
            pos = np.where(submask) 
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            finalmasks.append(submask)
            bounds.append([xmin, ymin, xmax, ymax]) # pascal_voc format
            class_ids.append(id)
        finalmasks = np.array(finalmasks)
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

        if self.to_tensor:
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
    
    def export_coco_json_builtin(self, json_path):
        coco = coco_utils.convert_to_coco_api(self)
        with open(json_path, 'w') as f:
            json.dump(coco.dataset, f)
        print('Exported coco json to: {0}'.format(json_path))
        return coco

    def export_coco_json(self, json_path, dataset: torch.utils.data.Dataset = None):
        def convert_classes_to_coco(classes):
            categories = []
            for i, c in enumerate(classes):
                category = {
                    'id': i,
                    'name': c,
                }
                categories.append(category)
            return categories
        
        def close_contour(contour):
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            return contour

        def binary_mask_to_polygon(binary_mask, tolerance=0):
            """Converts a binary mask to COCO polygon representation
            Args:
                binary_mask: a 2D binary numpy array where '1's represent the object
                tolerance: Maximum distance from original points of polygon to approximated
                    polygonal chain. If tolerance is 0, the original coordinate array is returned.
            """
            polygons = []
            # pad mask to close contours of shapes which start and end at an edge
            padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_binary_mask, 0.5)
            contours = np.subtract(contours, 1)
            for contour in contours:
                contour = close_contour(contour)
                contour = measure.approximate_polygon(contour, tolerance)
                if len(contour) < 3:
                    continue
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                # after padding and subtracting 1 we may get -0.5 points in our segmentation 
                segmentation = [0 if i < 0 else i for i in segmentation]
                polygons.append(segmentation)

            return polygons

        def convert_bbox_to_coco(bbox):
            # Returns [x, y, width, height] from pascal_voc format [x_min, y_min, x_max, y_max]
            return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer) :
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    v = obj.detach().cpu().numpy()
                    if v.shape == (1,):
                        return v.item()
                    elif v.ndim > 1:
                        return v.squeeze()
                    else:
                        return v
                return super(NpEncoder, self).default(obj)

        coco_images = []
        coco_anns = []
        iterable = self if dataset is None else dataset
        transform_state = []
        if isinstance(iterable, torch.utils.data.ConcatDataset):
            for dataset in iterable.datasets:
                transform_state.append((dataset.to_tensor, dataset.transforms))
                dataset.to_tensor = False
                dataset.transforms = False
        else:
            transform_state.append((iterable.to_tensor, iterable.transforms))
            iterable.to_tensor = False
            iterable.transforms = False
        for i in range(len(iterable)):
            print('Exporting image {0}/{1}'.format(i + 1, len(iterable)))
            img, target = iterable[i]
            filename = str(iterable.imgs[i]).split('/')[-1] if dataset is None else str(iterable.get_img(i)).split('/')[-1]
            img_id = target['image_id'].item() if dataset is None else i
            coco_image = {
                'id': img_id,
                'width': img.size[0],
                'height': img.size[1],
                'file_name': filename
            }
            for i in range(target['masks'].shape[0]):
                coco_ann = {
                    'id': len(coco_anns) + 1,
                    'image_id': img_id,
                    'category_id': target['labels'][i].item() - 1,
                    'segmentation': binary_mask_to_polygon(target['masks'][i]),
                    'area': target['area'][i].item(),
                    'bbox': convert_bbox_to_coco(target['boxes'][i].tolist()),
                    'iscrowd': target['iscrowd'][i].item()
                }
                coco_anns.append(coco_ann)
            coco_images.append(coco_image)
        

        coco_json = {
            'images': coco_images,
            'annotations': coco_anns,
            'categories': convert_classes_to_coco(LesionDataset.grading_classes())
        }
        
        # Create path of export folder if it does not exist
        export_folder = Path(json_path).parent
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        with open(json_path, 'w') as f:
            json.dump(coco_json, f, cls=NpEncoder)
        print('Exported coco json to: {0}'.format(json_path))
        if isinstance(iterable, torch.utils.data.ConcatDataset):
            for idx in range(len(iterable.datasets)):
                iterable.datasets[idx].to_tensor = transform_state[idx][0]
                iterable.datasets[idx].transforms = transform_state[idx][1]
        else:
            iterable.to_tensor = transform_state[0][0]
            iterable.transforms = transform_state[0][1]
        return coco_json
       
    
class ConcatLesionDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, to_tensor=True, transforms=True):
        super().__init__(datasets)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, target

    def get_img(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].imgs[sample_idx]

    def export_coco_json(self, json_path):
        self.datasets[0].export_coco_json(json_path, dataset=self)
