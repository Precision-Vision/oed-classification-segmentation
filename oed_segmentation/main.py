# IMPORTS
from functions import *


# IMPORT DATASET
dataset = LesionDataset(dataset_path, TRAIN_MODE, one_class_mode=one_class_mode, linecolors=linecolors, batch_no=batch_no, downsample=downsample, minimum_size=minimum_size, excluded_classes=excluded_classes)
test_dataset_01 = LesionDataset(dataset_path_test_01, TRAIN_MODE, one_class_mode=one_class_mode, linecolors=linecolors, batch_no=batch_no, downsample=downsample, minimum_size=minimum_size, excluded_classes=excluded_classes)
test_dataset_02 = LesionDataset(dataset_path_test_02, TRAIN_MODE, one_class_mode=one_class_mode, linecolors=linecolors, batch_no=batch_no, downsample=downsample, minimum_size=minimum_size, excluded_classes=excluded_classes)


# SPLIT DATASET
all_datasets = torch.utils.data.ConcatDataset([dataset, test_dataset_01])
train_dataset, validation_dataset = split_data(all_datasets, train_split_ratio, random_val)
# test_dataset_01 = split_data(test_dataset_01, [1.0, 0.0], random_val)[0]
test_dataset_02 = split_data(test_dataset_02, [1.0, 0.0], random_val)[0]
data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)
data_loader_validation = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)


# MODEL
model = get_model_instance_segmentation(num_classes)
model.to(device)


# TRAINING/INFERENCE
writer = SummaryWriter()
if TRAIN_MODE:
    print('Training model...')
    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate,
                                weight_decay=weight_decay)
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.01)

    for epoch in range(num_epochs):
        logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1000)
        print('logger', epoch, logger)
        writer.add_scalar('Loss/train', logger.meters['loss'].avg, epoch)
        writer.add_scalar('Loss/train/box_reg_loss', logger.meters['loss_box_reg'].avg, epoch)
        writer.add_scalar('Loss/train/mask', logger.meters['loss_mask'].avg, epoch)
        writer.add_scalar('Loss/train/obj_loss', logger.meters['loss_objectness'].avg, epoch)
        writer.add_scalar('Loss/train/cls_loss', logger.meters['loss_classifier'].avg, epoch)
        
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        eval1 = evaluate(model, data_loader_validation, device=device)
        dice, f1, acc = evaluate_plus(model, validation_dataset, device=device, max_predictions=max_detections, make_fig=False)
        mean_dice, mean_f1, mean_acc = sum(dice) / len(dice), sum(f1) / len(f1), sum(acc) / len(acc)
        bbox_stats = eval1.coco_eval['bbox'].stats
        mask_stats = eval1.coco_eval['segm'].stats
        write_stats(writer, bbox_stats, mask_stats, mean_dice, mean_f1, mean_acc, epoch)

        # Save model snapshot to file
        if not os.path.exists(model_path + 'snapshot/'):
            os.makedirs(model_path + 'snapshot/')
        torch.save(model.state_dict(), model_path + 'snapshot/' + 'model-epoch-' + str(epoch) + '.pt')
        
    # Save model to file
    torch.save(model.state_dict(), model_path + 'model.pt')
    print("Training complete")
else:
    print('Loading model from file...')
    # Load model from file
    model.load_state_dict(torch.load(model_path + model_filename))
    print("Model loaded")


# EVALUATION
avg_dices = []
mean_avg_f1_bboxes = []
mean_avg_acc_bboxes = []
for label, dataset in {'val': validation_dataset, 
                    #    'test01': test_dataset_01, 
                       'test02': test_dataset_02}.items():
    print('Evaluating model on {} items...\n\n'.format(len(dataset.indices)))
    for i in range(eval_times):
        dices, f1s, acc, figs = evaluate_plus(model, dataset, device, max_predictions=max_detections)
        avg_dice = sum(dices) / len(dices)
        avg_f1_bbox = sum(f1s) / len(f1s)
        avg_acc_bbox = sum(acc) / len(acc)
        avg_dices.append(avg_dice)
        mean_avg_f1_bboxes.append(avg_f1_bbox)
        mean_avg_acc_bboxes.append(avg_acc_bbox)
        writer.add_scalar(f"Dice/{label}", avg_dice, i)
        writer.add_scalar(f"F1/{label}", avg_f1_bbox, i)
        writer.add_scalar(f"Overlap Acc/{label}", avg_acc_bbox, i)
        print('{}: {}: Average Mask Dice: {}, Average BBox F1: {}, Average Overlap Acc: {} over {} items'.format(label, i, avg_dice, avg_f1_bbox, avg_acc_bbox, len(dataset.indices)))
        if i == 0:
            writer.add_figure(f"Prediction/{label}", figs)

    mean_avg_dice = sum(avg_dices) / len(avg_dices)
    mean_avg_f1_bbox = sum(mean_avg_f1_bboxes) / len(mean_avg_f1_bboxes)
    mean_avg_acc_bbox = sum(mean_avg_acc_bboxes) / len(mean_avg_acc_bboxes)
    print('{}: Mean Average Mask Dice: {}, Mean Average BBox F1: {}, Mean Average Overlap Acc: {} over {} items'.format(label, mean_avg_dice, mean_avg_f1_bbox, mean_avg_acc_bbox, len(dataset.indices)))
    writer.add_scalar(f"Dice/{label}/mean", mean_avg_dice, 0)
    writer.add_scalar(f"F1/{label}/mean", mean_avg_f1_bbox, 0)
    writer.add_scalar(f"Overlap Acc/{label}/mean", mean_avg_acc_bbox, 0)
writer.close()