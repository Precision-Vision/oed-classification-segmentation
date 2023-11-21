from config import *
import torchvision
import numpy as np
from LesionDataset import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import transforms as T
import torchvision.transforms as T2
from engine import train_one_epoch, evaluate
import utils
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib
from copy import deepcopy

plt.rcParams['figure.figsize'] = fig_size
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# SUPPORTING FUNCTIONS
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train:bool):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomIoUCrop())
        # transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)

def plot_transforms(imgs, orig_img, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def apply_mask(image, mask, class_id, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask,
                                image[:, :, c] *
                                (1 - alpha) + alpha * color[c],
                                image[:, :, c])
    return image

def apply_masks(image, masks, class_ids, colors, alpha=0.5,):
    """Apply the given mask to the image.
    """
    if masks.shape[0] == 0 or class_ids.shape[0] == 0:
        return image
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        class_id_str = str(np.array(class_ids[i]))
        color = colors[class_id_str][0]
        image = apply_mask(image, mask, class_ids[i], color, alpha)
    return image

def export_mask(img, target, filename):
    mask = target['masks']
    print('img', img.shape, type(img), img.dtype)
    print('masks', mask.shape, np.unique(mask), type(mask), mask.dtype)
    print('target boxes', target['boxes'], type(target['boxes']), target['boxes'].dtype)
    print('target area', target['area'], type(target['area']), target['area'].dtype)
    print('target labels', target['labels'], target['labels'].dtype)
    print('target image_id', target['image_id'], target['image_id'].dtype)
    # Save mask to file as CSV
    np.savetxt(filename, mask.cpu().numpy()[0], delimiter=',', fmt='%d')

def display_prediction(img, masks, boxes, colors={'1': [[1.0, 0.5, 0.0], 'rainbow', 'Dysplastic'], '2': [[0.0, 1.0, 0.0], 'Greens', 'Non-dysplastic']}):
    # Prepare img and mask
    img, masks = np.array(img), masks
    img = img.transpose((1, 2, 0))

    # Plot config
    fig, (ax0, ax1) = plt.subplots(2, 1)
    # Change figure size
    fig.set_figheight(5)
    fig.set_figwidth(2.5)
    ax0.axis('off')
    ax1.axis('off')
    ax0.imshow((img * 255).astype(np.uint8))
    ax1.imshow(img)
    ax0.title.set_text('Image')
    ax1.title.set_text('Prediction')
    if len(masks) == 0: return fig
    
    # Plot image and image with bbox and mask side by side
    sus_mask = masks[0].cpu().numpy()  
    sus_mask = sus_mask.transpose((1, 2, 0))
    for i in range(sus_mask.shape[2]):
        x1, y1, x2, y2 = boxes[0].cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
        alphas = matplotlib.colors.Normalize(0, .5, clip=True)(np.abs(sus_mask[:, :, i]))
        ax1.imshow(sus_mask[:, :, i], alpha=alphas, cmap=colors['1'][1])
        ax1.add_patch(rect)

    return fig

def display_instance(img, idx, masks, labels, boxes, dataset: LesionDataset, prediction=None, metrics=None, colors={'1': [[1.0, 0.5, 0.0], 'rainbow', 'Dysplastic'], '2': [[0.0, 1.0, 0.0], 'Greens', 'Non-dysplastic']}, max_predictions=None, grading_class=None):
    # Prepare img and mask
    img, mask, class_ids = np.array(img), masks, labels
    img = img.transpose((1, 2, 0))
    mask = np.array(mask).transpose((1, 2, 0)) if mask.shape[0] > 0 else np.array([])
    if isinstance(dataset, torch.utils.data.dataset.ConcatDataset):
        dataset = dataset.datasets[0] if idx < len(dataset.datasets[0]) else dataset.datasets[1] 

    # Plot config
    masks_count = mask.shape[2] if mask.shape[0] > 0 else 0
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    ax0.imshow((img * 255).astype(np.uint8))
    if grading_class is not None:
        grading_class_label = dataset.grading_classes[grading_class]
        ax0.title.set_text(grading_class_label)
    ax1.title.set_text('Annotation')

    # Plot image + mask
    img_masked = apply_masks(np.array(img), mask, class_ids, colors)
    ax1.imshow(img_masked)
    
    # Show bounding boxes of each mask
    for i in range(masks_count):
        x1, y1, x2, y2 = boxes[i]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
        ax1.add_patch(rect)

    # Prediction
    if prediction: 
        ax2.title.set_text('Prediction')
        if max_predictions is None: max_predictions = masks_count
        ax2.imshow(img)
        # Plotting prediction  
        prediction = prediction[0]
        pred_labels = prediction['labels'].cpu().numpy()
        sus_idx = np.where(pred_labels == dataset.binary_classes_indicies[0])[0]
        norm_idx = np.where(pred_labels == dataset.binary_classes_indicies[1])[0]
        
        # Normal predictions
        for i, idx in enumerate(norm_idx):
            if i >= max_predictions: break
            norm_mask = prediction['masks'][idx].cpu().numpy()  
            norm_mask = norm_mask.transpose((1, 2, 0))
            for i in range(norm_mask.shape[2]):
                if i >= max_predictions: break
                alphas = matplotlib.colors.Normalize(0, .5, clip=True)(np.abs(norm_mask[:, :, i]))
                ax2.imshow(norm_mask[:, :, i], alpha=alphas, cmap=colors['2'][1])
                x1, y1, x2, y2 = prediction['boxes'][idx].cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
                # ax2.text(x1, y1, colors['2'][2], color='black', fontsize=10)
                ax2.add_patch(rect)

        # Suspicious predictions
        for i, idx in enumerate(sus_idx):
            if i >= max_predictions: break
            sus_mask = prediction['masks'][idx].cpu().numpy()  
            sus_mask = sus_mask.transpose((1, 2, 0))
            for i in range(sus_mask.shape[2]):
                x1, y1, x2, y2 = prediction['boxes'][idx].cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
                alphas = matplotlib.colors.Normalize(0, .5, clip=True)(np.abs(sus_mask[:, :, i]))
                ax2.imshow(sus_mask[:, :, i], alpha=alphas, cmap=colors['1'][1])
                if metrics is not None:
                    for i, (key, value) in enumerate(metrics.items()):
                        if i == 0:
                            continue
                            # ax0.title.set_text('{}: {:.4f}'.format(key, value))
                        elif i == 1:
                            ax1.title.set_text('{}: {:.4f}'.format(key, value))
                        elif i == 2:
                            ax2.title.set_text('{}: {:.4f}'.format(key, value))
                        else:
                            break
                ax2.add_patch(rect)
            
        return fig
        
    else:
        # Remove ax2 from plt
        ax2.set_visible(False)

    plt.show()
    plt.tight_layout()

def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = torch.mean((2. * intersection + smooth) / (union + smooth))
    return dice

def compute_dice(y_trues, y_preds, smooth=1):
    mean_dices = []
    for i, y_true in enumerate(y_trues):
        dices = []
        for y_pred in y_preds:
            y_pred = torch.squeeze(y_pred, dim=0)
            dice = dice_coef(y_true, y_pred)
            dices.append(dice)
        # Compute the mean dice coefficient
        if len(dices) > 0:
            mean_dices.append(torch.mean(torch.stack(dices)))
    if len(mean_dices) > 0:
        mean_dice = torch.mean(torch.stack(mean_dices)) # Average of mean dice coefficients over all images' masks
    else:
        return 0
    return mean_dice

def sum_average_precision(stats):
    # Set any -1 values to 0
    stats = [0 if x == -1 else x for x in stats]
    return stats[0] + stats[1] + stats[2] + stats[3] + stats[4] + stats[5]

def sum_average_recall(stats):
    # Set any -1 values to 0
    stats = [0 if x == -1 else x for x in stats]
    return stats[6] + stats[7] + stats[8] + stats[9] + stats[10] + stats[11]

def bbox_iou(y_true_bbox, y_pred_bbox, padding_allowance = 0.85):
    # Compute intersection
    x1 = max(y_true_bbox[0], y_pred_bbox[0])
    y1 = max(y_true_bbox[1], y_pred_bbox[1])
    x2 = min(y_true_bbox[2], y_pred_bbox[2])
    y2 = min(y_true_bbox[3], y_pred_bbox[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # Compute union
    y_true_area = (y_true_bbox[2] - y_true_bbox[0] + 1) * (y_true_bbox[3] - y_true_bbox[1] + 1)
    y_pred_area = (y_pred_bbox[2] - y_pred_bbox[0] + 1) * (y_pred_bbox[3] - y_pred_bbox[1] + 1)
    union = y_true_area + y_pred_area - intersection
    # Compute IOU
    iou = intersection / (union * padding_allowance)
    return iou

def bbox_ious(y_trues_bbox, y_preds_bbox):
    ious = []
    for i, y_true_bbox in enumerate(y_trues_bbox):
        # Check if index is out of range
        if i >= len(y_preds_bbox):
            break
        iou = bbox_iou(y_true_bbox, y_preds_bbox[i])
        ious.append(iou)
    return ious

def compute_overlap_accuracy_bbox(y_trues_bbox, y_preds_bbox, overlap_threshold=0.5):
    acc = []
    for iou in bbox_ious(y_trues_bbox, y_preds_bbox):
        if iou >= overlap_threshold:
            acc.append(torch.tensor(1, dtype=torch.float32))
        else:
            acc.append(iou.cpu())
    if len(acc) == 0:
        return 0
    mean_acc = torch.mean(torch.stack(acc))
    return mean_acc

def compute_f1_bbox(y_trues_bbox, y_preds_bbox):
    f1s = []
    for i, iou in enumerate(bbox_ious(y_trues_bbox, y_preds_bbox)):
        f1 = 2 * (iou) / (1 + iou)
        f1s.append(f1)
    # Compute mean F1
    if len(f1s) == 0:
        return 0
    mean_f1 = torch.mean(torch.stack(f1s))
    return mean_f1

def split_data(dataset, train_split_ratio=[0.80, 0.20], random_val=False):
    if random_val:
        dataset_length = len(dataset)
        train_split = int(dataset_length * train_split_ratio[0])
        validation_split = int(dataset_length * train_split_ratio[1])
        if train_split + validation_split < dataset_length:
            validation_split += dataset_length - (train_split + validation_split)
        elif train_split + validation_split > dataset_length:
            validation_split -= (train_split + validation_split) - dataset_length
        
        random_generator = torch.Generator().manual_seed(42)
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_split, validation_split], generator=random_generator)
    else: 
        validation_indices = [1, 2, 10, 11, 22, 29]
        training_indices = [i for i in range(len(dataset)) if i not in validation_indices]
        train_dataset = torch.utils.data.Subset(dataset, training_indices)
        validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    return train_dataset, validation_dataset

def write_stats(writer, bbox_stats, mask_stats, dice, f1, acc, epoch):
    writer.add_scalar('Eval/ap_bbox(IoU=0.50:0.95|area=all|maxDets=100)', bbox_stats[0], epoch)
    writer.add_scalar('Eval/ap_bbox(IoU=0.50|area=all|maxDets=100)', bbox_stats[1], epoch)
    writer.add_scalar('Eval/ap_bbox(IoU=0.75|area=all|maxDets=100)', bbox_stats[2], epoch)
    writer.add_scalar('Eval/ap_bbox(IoU=0.50:0.95|area=small|maxDets=100)', bbox_stats[3], epoch)
    writer.add_scalar('Eval/ap_bbox(IoU=0.50:0.95|area=medium|maxDets=100)', bbox_stats[4], epoch)
    writer.add_scalar('Eval/ap_bbox(IoU=0.50:0.95|area=large|maxDets=100)', bbox_stats[5], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=all|maxDets=1)', bbox_stats[6], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=all|maxDets=10)', bbox_stats[7], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=all|maxDets=100)', bbox_stats[8], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=small|maxDets=100)', bbox_stats[9], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=medium|maxDets=100)', bbox_stats[10], epoch)
    writer.add_scalar('Eval/ar_bbox(IoU=0.50:0.95|area=large|maxDets=100)', bbox_stats[11], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.50:0.95|area=all|maxDets=100)', mask_stats[0], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.50|area=all|maxDets=100)', mask_stats[1], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.75|area=all|maxDets=100)', mask_stats[2], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.50:0.95|area=small|maxDets=100)', mask_stats[3], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.50:0.95|area=medium|maxDets=100)', mask_stats[4], epoch)
    writer.add_scalar('Eval/ap_mask(IoU=0.50:0.95|area=large|maxDets=100)', mask_stats[5], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=all|maxDets=1)', mask_stats[6], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=all|maxDets=10)', mask_stats[7], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=all|maxDets=100)', mask_stats[8], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=small|maxDets=100)', mask_stats[9], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=medium|maxDets=100)', mask_stats[10], epoch)
    writer.add_scalar('Eval/ar_mask(IoU=0.50:0.95|area=large|maxDets=100)', mask_stats[11], epoch)
    writer.add_scalar('Eval/dice_mask', dice, epoch)
    writer.add_scalar('Eval/f1_bbox', f1, epoch)
    writer.add_scalar('Eval/overlap_acc_bbox', acc, epoch)
    writer.flush()

def evaluate2(model, validation_dataset, dataset, device):
    # Prepare data and compute predictions
    y_trues = []
    y_preds = []
    for idx in validation_dataset.indices:
        img, target = dataset[idx]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
        
        y_trues = target['masks'].to(device)
        y_preds = prediction[0]['masks']
    
    # Compute metrics
    metrics = {}
    if len(y_trues) > 0 and len(y_preds) > 0:
        dice = compute_dice(y_trues, y_preds)
        metrics['dice'] = dice
    else:
        metrics['dice'] = 0
    return metrics

def eval_one_image_external(model, img, device):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    mask = prediction[0]['masks']
    labels = prediction[0]['labels']
    boxes = prediction[0]['boxes']
    fig = display_prediction(img, mask, boxes)
    return fig

def eval_one_image(model, dataset, idx, device):
    img, target = dataset[idx]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    default_colors = dataset.class_colors
    colors = {
        '1': default_colors['Cancer'] if 'Dysplastic' in dataset.excluded_classes else default_colors['Dysplastic'],
        '2': default_colors['Non-dysplastic'],
    }
    fig = display_instance(img, idx, target['masks'], target['labels'], target['boxes'], dataset, prediction, colors=colors, grading_class=target['grading_class'].cpu())
    return fig

def evaluate_plus(model, subset, device, max_predictions=None, make_fig=True):
    dices = []
    f1s = []
    acc = []
    figs = []
    dataset = subset.dataset
    for idx in subset.indices:
        # print(torch.cuda.memory_stats())
        img, target = dataset[idx]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
        
        y_trues = target['masks'].to(device)
        y_preds = prediction[0]['masks']
        if max_predictions is not None and len(y_preds) >= max_predictions and len(y_trues) >= max_predictions:
            y_trues = y_trues[:max_predictions]
            y_preds = y_preds[:max_predictions]
        elif max_predictions is None and len(y_trues) > len(y_preds):
            y_preds = y_preds[:len(y_trues)]

        y_trues_bbox = target['boxes'].to(device)
        y_preds_bbox = prediction[0]['boxes']
        if max_predictions is not None and len(y_preds_bbox) >= max_predictions and len(y_trues_bbox) >= max_predictions:
            y_trues_bbox = y_trues_bbox[:max_predictions]
            y_preds_bbox = y_preds_bbox[:max_predictions]
        elif max_predictions is None and len(y_trues_bbox) > len(y_preds_bbox):
            y_preds_bbox = y_preds_bbox[:len(y_trues_bbox)]

        dice = compute_dice(y_trues, y_preds)
        f1_bbox = compute_f1_bbox(y_trues_bbox, y_preds_bbox)
        acc_bbox = compute_overlap_accuracy_bbox(y_trues_bbox, y_preds_bbox, overlap_threshold=overlap_threshold)
        dices.append(dice)
        f1s.append(f1_bbox)
        acc.append(acc_bbox)
        print('{}: Mask Dice: {}, BBox F1: {}, BBox Overlap Acc: {}'.format(idx, dice, f1_bbox, acc_bbox))

        if make_fig: 
            _dataset = dataset.datasets[0] if isinstance(dataset, torch.utils.data.dataset.ConcatDataset) else dataset
            default_colors = _dataset.class_colors
            colors = {
                '1': default_colors['Cancer'] if 'Dysplastic' in _dataset.excluded_classes else default_colors['Dysplastic'],
                '2': default_colors['Non-dysplastic'],
            }
            metrics = {
                'Dice': dice,
                'F1': f1_bbox,
                'Accuracy': acc_bbox
            }
            fig = display_instance(img, idx, target['masks'], target['labels'], target['boxes'], dataset, prediction, metrics, colors=colors, max_predictions=max_predictions, grading_class=target['grading_class'])
            figs.append(fig)    
        if not PRED_PLOT:
            break
    output = (dices, f1s, acc, figs) if make_fig else (dices, f1s, acc)
    return output

def eval_map_mar(model, data_loader_test, device):
    eval = evaluate(model, data_loader_test, device=device)
    bbox_stats = eval.coco_eval['bbox'].stats
    mask_stats = eval.coco_eval['segm'].stats
    map_bbox = sum_average_precision(bbox_stats)
    mar_bbox = sum_average_recall(bbox_stats)
    map_mask = sum_average_precision(mask_stats)
    mar_mask = sum_average_recall(mask_stats)
    print('bbox_stats', bbox_stats) 
    print('mask_stats', mask_stats)
    print('map_bbox', map_bbox)
    print('mar_bbox', mar_bbox)
    print('map_mask', map_mask)
    print('mar_mask', mar_mask)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
