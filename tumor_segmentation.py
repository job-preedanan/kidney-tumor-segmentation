import numpy as np
import os
import cv2
import tensorflow as tf
import random
import util_function as utils
# from unet_based_model import UNet
from pretrained_Unet import get_pretrained_unet
# from MulmoNet_vgg_pretrained import make_mulmoXNet_vgg16
import loss_function as custom_loss
from data_loader import load_data_from_xlsx
from skimage.measure import label
from skimage.morphology import disk, dilation
import albumentations as A

PATH = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor'
# PATH = '/kw_resources/kidney_tumor'
DATA_FOLDER = '/dataset/'
EXPORT_FOLDER = '/exports/segmentation/crop_kidney/pretrained_unet/iw_dice/'
# EXPORT_FOLDER = '/exports/segmentation/pretrained_unet/'
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCH = 150
CROP_RATIO = 0
CROSS_NUM = 0


# list moving function
def list_index_move(list, split_num):
    split_idx = int(round(len(list) * split_num))  # split index
    new_list = list[split_idx:]
    new_list = np.concatenate([new_list, list[:split_idx]])
    return new_list


# histogram equalization + cropping + resize + normalize
def preprocessing(image, histeq=True):
    # hist equalization
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    crop_size = round(image.shape[1] * CROP_RATIO)
    image = image[:, crop_size:image.shape[1] - crop_size]  # width cropping
    image = np.array(utils.normalize_x(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))), np.float32)  # normalize(-1, 1)

    return image


# data spliting function
def split_train_test(x, y, val_ratio=0.2, random_sample=True):
    # zip x and y
    samples = list(zip(x, y))
    if random_sample:
        random.shuffle(samples)

    split_idx = int(round(len(samples) * val_ratio))  # split index
    test = samples[:split_idx]
    train = samples[split_idx:]

    # unzip and convert to array
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


def cc2weight_batch(batch_images, w_min: float = 1., w_max: float = 100., bias=1.0):
    images = np.zeros((batch_images.shape[2], batch_images.shape[1], batch_images.shape[0]))
    cc_batch = np.zeros_like(images)

    # dilate contour + find contour in batch
    total_cc = 0
    total_cc_area = 0
    for i in range(batch_images.shape[0]):
        images[:, :, i] = dilation(np.squeeze(batch_images[i]), selem=disk(1))
        cc = label(images[:, :, i], background=0, connectivity=2)  # find cc in an image
        cc[cc != 0] += total_cc
        total_cc += len(np.unique(cc)) - 1  # not count bg cc (==0)
        cc_batch[:, :, i] = cc
        total_cc_area += np.prod(cc.shape) - len(cc[cc == 0])  # total contour area (no bg)

    # inverse weighting calculation
    cc_items = np.unique(cc_batch)[1:]  # only stone contours
    weight = np.ones_like(images, dtype='float32')
    #
    K = len(cc_items)
    for j in cc_items:
        iw1 = bias + (total_cc_area / (K * np.sum(cc_batch == j)))
        iw2 = total_cc_area / (K * np.sum(cc_batch == j))
        weight[cc_batch == j] = iw1
        # print('cnt[' + str(int(i)) + '] size = ' + str(np.sum(cc_batch == i)) + ', iw = ' + str(iw2))

    weight = np.clip(weight, w_min, w_max)

    # append w_map to batch
    batch_ccweight = np.ones_like(batch_images)
    for i in range(batch_ccweight.shape[0]):
        batch_ccweight[i] = weight[:, :, i][:, :, np.newaxis]

    return batch_ccweight


def augmentation(modal_batch_images, modal_batch_masks):

    # define augmentation methods
    transform = A.Compose([A.HorizontalFlip(p=0.5),
                           A.VerticalFlip(p=0.5),
                           A.Rotate(p=0.5, limit=(-20, 20))])

    aug_batch_images = np.ones((len(modal_batch_images), BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, modal_batch_images[0].shape[-1]))
    aug_batch_masks = np.ones((len(modal_batch_masks), BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, modal_batch_masks[0].shape[-1]))

    # modal input
    for m in range(len(modal_batch_images)):
        batch_images = modal_batch_images[m]
        batch_masks = modal_batch_masks[m]

        # batch
        for b in range(len(batch_images)):
            augmented = transform(image=batch_images[b], mask=batch_masks[b])
            aug_batch_images[m, b] = augmented['image']
            aug_batch_masks[m, b] = augmented['mask']


    #     if b < 6:
    #         # display sample images
    #         axarr[b, 0].grid(False)
    #         axarr[b, 0].imshow(utils.denormalize_x(batch_images[b]), cmap='gray', vmin=0, vmax=255)
    #         axarr[b, 1].grid(False)
    #         axarr[b, 1].imshow(utils.denormalize_x(aug_batch_images[b]), cmap='gray', vmin=0, vmax=255)
    #         axarr[b, 2].grid(False)
    #         axarr[b, 2].imshow(utils.denormalize_y(batch_masks[b]), cmap='gray', vmin=0, vmax=255)
    #         axarr[b, 3].grid(False)
    #         axarr[b, 3].imshow(utils.denormalize_y(aug_batch_masks[b]), cmap='gray', vmin=0, vmax=255)
    #
    # plt.show()
    return aug_batch_images, aug_batch_masks


def datagenerator(images, masks, batch_size):

    while True:
        start = 0
        end = batch_size

        while start < len(images[0]):

            # check #modal input

            # ------------------ 2 modals input --------------------------------
            if len(images) == 2:
                # load batch images + masks
                x1 = images[0][start:end]
                y1 = masks[0][start:end]
                x2 = images[1][start:end]
                y2 = masks[1][start:end]

                # apply augmentation
                aug_x, aug_y = augmentation([x1, x2], [y1, y2])
                aug_x1, aug_x2 = aug_x[0], aug_x[1]
                aug_y1, aug_y2 = aug_y[0], aug_y[1]

                # create iw map
                iw_map = cc2weight_batch(y)
                aug_y = np.concatenate((aug_y, iw_map), axis=-1)

                yield [aug_x], [aug_y]

            # ------------------ 1 modal input --------------------------------
            else:
                # load batch images + masks
                x = images[0][start:end]
                y = masks[0][start:end]

                # apply augmentation
                aug_x, aug_y = augmentation([x], [y])
                aug_x, aug_y = aug_x[0], aug_y[0]

                # create iw map
                iw_map = cc2weight_batch(aug_y)
                aug_y = np.concatenate((aug_y, iw_map), axis=-1)

                # display sample images
                # f, axarr = plt.subplots(6, 3)
                # for b in range(6):
                #     axarr[b, 0].grid(False)
                #     axarr[b, 0].imshow(utils.denormalize_x(aug_x[b]), cmap='gray', vmin=0, vmax=255)
                #     axarr[b, 1].grid(False)
                #     axarr[b, 1].imshow(utils.denormalize_y(aug_y[b, :, :, 0]), cmap='gray', vmin=0, vmax=255)
                #     axarr[b, 2].grid(False)
                #     axarr[b, 2].imshow(255*aug_y[b, :, :, 1]/(np.max(aug_y[:, :, :, 1]) - np.min(aug_y[:, :, :, 1])),
                #                        cmap='gray', vmin=0, vmax=255)
                #
                # plt.show()

                yield aug_x, aug_y

            start += batch_size
            end += batch_size


def train(x_train, y_train, x_val, y_val, modal_name):
    modal_names = modal_name[0]
    for i in range(len(modal_name) - 1):
        modal_names += '_' + modal_name[i + 1]

    print(modal_names)

    try:
        # os.makedirs(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM))
        os.makedirs(PATH + EXPORT_FOLDER + modal_names)
    except(FileExistsError):
        print('folders exist')

    # split modals
    x_train1 = x_train[:, :, :, :3]
    # x_train2 = x_train[:, :, :, 3:6]
    # x_train3 = x_train[:, :, :, 6:]
    x_val1 = x_val[:, :, :, :3]
    # x_val2 = x_val[:, :, :, 3:6]
    # x_val3 = x_val[:, :, :, 6:]
    y_train1 = np.expand_dims(y_train[:, :, :, 0], axis=-1)
    # y_train2 = np.expand_dims(y_train[:, :, :, 1], axis=-1)
    y_val1 = np.expand_dims(y_val[:, :, :, 0], axis=-1)
    # y_val2 = np.expand_dims(y_val[:, :, :, 1], axis=-1)
    y_val1 = np.concatenate((y_val1, np.ones_like(y_val1)), axis=-1)

    # load U-Net network
    # model = UNet(3, 1, 32).get_model()
    model = get_pretrained_unet(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
    model.summary()

    model.compile(loss=custom_loss.iw_dice_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[custom_loss.dice_coef, custom_loss.recall, custom_loss.precision])

    # data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                          width_shift_range=0,
                                                          height_shift_range=0,
                                                          zoom_range=False,
                                                          fill_mode='nearest',
                                                          vertical_flip=False,
                                                          horizontal_flip=True)

    # generator flow (3 modal inputs)
    def generator(x_input, y_input, batch_size):

        if len(x_input) == 2:
            x1 = x_input[0]
            x2 = x_input[1]
        elif len(x_input) == 3:
            x1 = x_input[0]
            x2 = x_input[1]
            x3 = x_input[3]
        else:
            x1 = x_input[0]

        if len(y_input) == 2:
            y1 = y_input[0]
            y2 = y_input[1]
        elif len(y_input) == 3:
            y1 = y_input[0]
            y2 = y_input[1]
            y3 = y_input[3]
        else:
            y1 = y_input[0]

        gen_1 = aug.flow(x1, y1, batch_size=batch_size, seed=1)
        # gen_2 = aug.flow(x2, y2, batch_size=batch_size, seed=1)
        # gen_3 = aug.flow(x3, y, batch_size=batch_size, seed=1)
        while True:
            x1 = gen_1.next()
            # x2 = gen_2.next()
            # x3 = gen_3.next()

            # ((x1, x2, x3), y1)
            # yield [x1[0], x2[0]], [x1[1], x2[1]]
            yield [x1[0]], [x1[1]]

    # learning rate decay callback
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    # model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=PATH + EXPORT_FOLDER + modal_names + '/weight_checkpoint.hdf5',   #  + '/cross_val#' + str(CROSS_NUM)
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    # fits the model on batches with real-time data augmentation:
    # history = model.fit(generator([x_train1, x_train2], [y_train1, y_train2], batch_size=BATCH_SIZE),
    #                     steps_per_epoch=int(len(x_train1) / BATCH_SIZE),
    #                     epochs=EPOCH,
    #                     callbacks=[lr_reduce_callback, model_checkpoint_callback],
    #                     validation_data=([x_val1, x_val2], [y_val1, y_val2]))

    history = model.fit(datagenerator(x_train1, y_train1, batch_size=BATCH_SIZE),
                        steps_per_epoch=int(len(x_train1) / BATCH_SIZE),
                        epochs=EPOCH,
                        callbacks=[lr_reduce_callback, model_checkpoint_callback],
                        validation_data=([x_val1], [y_val1]))

    # model.save_weights(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM) + '/model_weights.hdf5')
    # utils.plot_summary_graph(history, PATH + EXPORT_FOLDER + '/' + modal_names + '/cross_val#' + str(CROSS_NUM))
    model.save_weights(PATH + EXPORT_FOLDER + modal_names + '/model_weights.hdf5')
    utils.plot_summary_graph(history, PATH + EXPORT_FOLDER + '/' + modal_names)


def predict(x_test, y_test, modal_name):
    modal_names = modal_name[0]
    for i in range(len(modal_name) - 1):
        modal_names += '_' + modal_name[i + 1]

    print(modal_names)

    try:
        os.makedirs(PATH + EXPORT_FOLDER + modal_names)
        # os.makedirs(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM))
    except(FileExistsError):
        print('folders exist')

    def pixelbased_metric(y_true, y_pred, binary_th=0.5):
        _, y_true = cv2.threshold(y_true, binary_th * 255, 255, cv2.THRESH_BINARY)
        _, y_pred = cv2.threshold(y_pred, binary_th * 255, 255, cv2.THRESH_BINARY)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        tp = np.sum(np.logical_and(y_true == 255, y_pred == 255))
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 255))
        fn = np.sum(np.logical_and(y_true == 255, y_pred == 0))

        return tp, fn, fp

    def metrics_compute(tp, fn, fp):
        def recall_compute(tp, fn):
            return round((tp * 100) / (tp + fn), 2) if tp + fn > 0 else 0

        def precision_compute(tp, fp):
            return round((tp * 100) / (tp + fp), 2) if tp + fp > 0 else 0

        def f_score_compute(tp, fn, fp):
            a = 2 * recall_compute(tp, fn) * precision_compute(tp, fp)
            b = recall_compute(tp, fn) + precision_compute(tp, fp)
            return round((a / b), 2) if b > 0 else 0

        return recall_compute(tp, fn), precision_compute(tp, fp), f_score_compute(tp, fn, fp)

    # split modals
    x_test1 = x_test[:, :, :, :3]
    # x_test2 = x_test[:, :, :, 3:6]
    # x_test3 = x_test[:, :, :, 6:]

    y_test1 = np.expand_dims(y_test[:, :, :, 0], axis=-1)
    # y_test2 = np.expand_dims(y_test[:, :, :, 1], axis=-1)

    # load U-Net network with trained weights
    model = get_pretrained_unet(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
    # model.load_weights(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM) + '/weight_checkpoint.hdf5')
    # model = UNet(3, 1, 32).get_model()
    model.load_weights(PATH + EXPORT_FOLDER + modal_names + '/model_weights.hdf5')

    # predict
    y_preds1 = model.predict([x_test1], batch_size=BATCH_SIZE)
    # [y_preds1, y_preds2] = model.predict([x_test1, x_test2], batch_size=BATCH_SIZE)

    # evaluation (find pixel-wise recall, precision, f1 score)
    recall = []
    precision = []
    f1_score = []
    try:
        # os.makedirs(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM) + '/results')
        os.makedirs(PATH + EXPORT_FOLDER + modal_names + '/results')
    except(FileExistsError):
        print('folders exist')

    for i, y_pred in enumerate(y_preds1):
        # denormalize (convert to 8-bit grayscale)
        y_true = np.array(utils.denormalize_y(y_test1[i, :, :, 0]), dtype=np.uint8)  # [i, :, :, 0]
        y_pred = np.array(utils.denormalize_y(y_pred[:, :, 0]), dtype=np.uint8)

        # pixel-based
        tp, fn, fp = pixelbased_metric(y_true, y_pred, binary_th=0.5)

        recall.append(metrics_compute(tp, fn, fp)[0])
        precision.append(metrics_compute(tp, fn, fp)[1])
        f1_score.append(metrics_compute(tp, fn, fp)[2])

        # save heatmap comparison
        org_image = utils.denormalize_x(x_test1[i, :, :, 0])
        y_true_heatmap = utils.convert_to_heatmap(org_image, y_true)
        y_pred_heatmap = utils.convert_to_heatmap(org_image, y_pred)

        y_true_save = np.stack((y_true,) * 3, axis=-1)
        y_pred_save = np.stack((y_pred,) * 3, axis=-1)
        save_image_top = np.concatenate((y_true_save, y_pred_save), axis=1)
        save_image_bot = np.concatenate((y_true_heatmap, y_pred_heatmap), axis=1)
        save_image = np.concatenate((save_image_top, save_image_bot), axis=0)
        # cv2.imwrite(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM) + '/results/' + str(i) + '_f1_' + str(
        #     round(metrics_compute(tp, fn, fp)[2])) + '.png', save_image)
        cv2.imwrite(PATH + EXPORT_FOLDER + modal_names + '/results/' + str(i) + '_f1_' + str(
            round(metrics_compute(tp, fn, fp)[2])) + '.png', save_image)

    def average(lst):
        return sum(lst) / len(lst)

    print('Recall: ' + str(average(recall)))
    print('Precision: ' + str(average(precision)))
    print('F1 score: ' + str(average(f1_score)))

    with open(PATH + EXPORT_FOLDER + modal_names + '/results_model_weights.txt', 'w') as f:
    # with open(PATH + EXPORT_FOLDER + modal_names + '/cross_val#' + str(CROSS_NUM) + '/results.txt', 'w') as f:
        f.write('Recall: ' + str(average(recall)))
        f.write('\n')
        f.write('Precision: ' + str(average(precision)))
        f.write('\n')
        f.write('F1 score: ' + str(average(f1_score)))

    # # second modal
    # recall = []
    # precision = []
    # f1_score = []
    # try:
    #     os.makedirs(PATH + EXPORT_FOLDER + modal_names + '/results2')
    # except(FileExistsError):
    #     print('folders exist')
    #
    # for i, y_pred in enumerate(y_preds2):
    #     # denormalize (convert to 8-bit grayscale)
    #     y_true = np.array(utils.denormalize_y(y_test2[i, :, :, 0]), dtype=np.uint8)  # [i, :, :, 0]
    #     y_pred = np.array(utils.denormalize_y(y_pred[:, :, 0]), dtype=np.uint8)
    #
    #     # pixel-based
    #     tp, fn, fp = pixelbased_metric(y_true, y_pred, binary_th=0.5)
    #
    #     recall.append(metrics_compute(tp, fn, fp)[0])
    #     precision.append(metrics_compute(tp, fn, fp)[1])
    #     f1_score.append(metrics_compute(tp, fn, fp)[2])
    #
    #     # save heatmap comparison
    #     org_image = utils.denormalize_x(x_test2[i, :, :, 0])
    #     y_true_heatmap = utils.convert_to_heatmap(org_image, y_true)
    #     y_pred_heatmap = utils.convert_to_heatmap(org_image, y_pred)
    #
    #     y_true_save = np.stack((y_true,) * 3, axis=-1)
    #     y_pred_save = np.stack((y_pred,) * 3, axis=-1)
    #     save_image_top = np.concatenate((y_true_save, y_pred_save), axis=1)
    #     save_image_bot = np.concatenate((y_true_heatmap, y_pred_heatmap), axis=1)
    #     save_image = np.concatenate((save_image_top, save_image_bot), axis=0)
    #     cv2.imwrite(PATH + EXPORT_FOLDER + modal_names + '/results2/' + str(i) + '_f1=' + str(
    #         round(metrics_compute(tp, fn, fp)[2])) + '.png', save_image)
    #
    # print('Recall: ' + str(average(recall)))
    # print('Precision: ' + str(average(precision)))
    # print('F1 score: ' + str(average(f1_score)))
    #
    # with open(PATH + EXPORT_FOLDER + modal_names + '/results.txt', 'w') as f:
    #     f.write('Recall: ' + str(average(recall)))
    #     f.write('\n')
    #     f.write('Precision: ' + str(average(precision)))
    #     f.write('\n')
    #     f.write('F1 score: ' + str(average(f1_score)))


if __name__ == '__main__':
    print(tf.__version__)
    import matplotlib.pyplot as plt

    modal_lists = [['pc'], ['ec'], ['dc'], ['tm'], ['am']]
    # modal_lists = [['pc', 'ec'], ['pc', 'dc'], ['pc', 'tm'], ['pc', 'am']]

    for modal_name in modal_lists:
        x_data, labels = load_data_from_xlsx(PATH + DATA_FOLDER, modal_lists=modal_name, image_type=1)
        y_data, _ = load_data_from_xlsx(PATH + DATA_FOLDER, modal_lists=modal_name, image_type=0)
        print(x_data.shape)
        print(y_data.shape)

        # # display sample images
        # f, axarr = plt.subplots(2, len(modal_lists))
        # idx = random.randint(0, len(x_data))
        # for m, modal in enumerate(modal_lists):
        #     axarr[0, m].set_title(modal)
        #     axarr[0, m].grid(False)
        #     axarr[0, m].imshow(utils.denormalize_x(x_data[idx, :, :, m*3:m*3 + 3]), cmap='gray', vmin=0, vmax=255)
        #     axarr[1, m].imshow(utils.denormalize_y(y_data[idx, :, :, m]), cmap='gray', vmin=0, vmax=255)
        #
        # plt.show()

        # for CROSS_NUM in range(5):
        #     print(CROSS_NUM)
        #     # split train - test
        #     x_train, y_train, x_test, y_test = split_train_test(x_data, y_data, val_ratio=0.2, random_sample=False)
        #
        #     print(x_train.shape)
        #     print(y_test.shape)
        #     print(x_train.shape)
        #     print(y_test.shape)
        #
        #     train(x_train, y_train, x_test, y_test, modal_name)
        #     predict(x_test, y_test, modal_name)
        #
        #     x_data = list_index_move(x_data, split_num=0.2)
        #     y_data = list_index_move(y_data, split_num=0.2)

        # split train - test
        x_train, y_train, x_test, y_test = split_train_test(x_data, y_data, val_ratio=0.2, random_sample=False)

        print(x_train.shape)
        print(y_test.shape)
        print(x_train.shape)
        print(y_test.shape)

        train(x_train, y_train, x_test, y_test, modal_name)
        predict(x_test, y_test, modal_name)