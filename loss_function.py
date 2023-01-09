import tensorflow as tf


# Recall
def recall(y_true, y_pred):
    y_true = tf.reshape(y_true[:, :, :, 0], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_true) + 1
    return intersection / denominator


# Precision
def precision(y_true, y_pred):
    y_true = tf.reshape(y_true[:, :, :, 0], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_pred) + 1
    return intersection / denominator


def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true[:, :, :, 0], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_true ** 2) + tf.reduce_sum(y_pred ** 2) + 1
    return 2.0 * intersection / denominator


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iw_dice_loss(y, y_pred):
    y_true = tf.reshape(y[:, :, :, 0], [-1])
    y_map = tf.reshape(y[:, :, :, 1], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_map * y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_map * (y_true ** 2)) + tf.reduce_sum(y_map * (y_pred ** 2)) + 1
    dice = 2.0 * intersection / denominator
    return 1.0 - dice


def focal_tv_loss(beta, gamma):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true * y_pred) + \
                      (1 - beta) * tf.reduce_sum((1 - y_true) * y_pred) + \
                      beta * tf.reduce_sum(y_true * (1 - y_pred))
        loss_value = (intersection + 1) / (denominator + 1)

        return tf.pow(1.0 - loss_value, (1/gamma))

    return loss


def iw_focal_tv_loss(beta, gamma):
    def loss(y, y_pred):
        y_true = tf.reshape(y[:, :, :, 0], [-1])
        y_map = tf.reshape(y[:, :, :, 1], [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred * y_map)
        denominator = tf.reduce_sum(y_true * y_pred * y_map) + \
                      (1 - beta) * tf.reduce_sum(y_map * (1 - y_true) * y_pred) + \
                      beta * tf.reduce_sum(y_map * y_true * (1 - y_pred))
        loss_value = (intersection + 1) / (denominator + 1)

        return tf.pow(1.0 - loss_value, (1/gamma))

    return loss


def combined_loss(y_true, y_pred):
    def dice(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return numerator / denominator

    alpha = 0.2
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    loss = alpha * tf.keras.losses.binary_crossentropy(y_true, y_pred) - (1 - alpha) * dice(y_true, y_pred)
    return loss


def iw_combined_loss(y, y_pred):
    def iw_dice(y_true, y_pred, y_map):
        numerator = 2 * tf.reduce_sum(y_true * y_pred * y_map)
        denominator = tf.reduce_sum(y_map * (y_true + y_pred))
        return numerator / denominator

    alpha = 0.2
    y_true = tf.reshape(y[:, :, :, 0], [-1])
    y_map = tf.reshape(y[:, :, :, 1], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    loss = alpha * tf.keras.losses.binary_crossentropy(y_true, y_pred) - (1 - alpha) * iw_dice(y_true, y_pred, y_map)
    return loss