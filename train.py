import time
import numpy as np
import os
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from model import Model
from dataloader import DataLoaderKITTI

def smooth_l1_loss(disps_pred, disps_targets, sigma=1.0):
    sigma_2 = sigma ** 2
    box_diff = disps_pred - disps_targets
    in_box_diff = box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
    return loss_box

parser = argparse.ArgumentParser()
parser.add_argument("--left", help="Dir path to left view")
parser.add_argument("--right", help="Dir path to right view")
parser.add_argument("--disp", help="Dir path to output heatmap")
parser.add_argument("--bs", default=8, type=int, help="Batch Size")
parser.add_argument("--maxdisp", default=128, type=int)
parser.add_argument("--epochs", default=10, type=int, help="No. of epochs")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--height", default=256, type=int)
parser.add_argument("--width", default=512, type=int)
parser.add_argument("--cnn_3d_type", default="resnet_3d", type=str)
parser.add_argument("--save", default="output", type=str, help="Path to model output")
parser.add_argument("--train_size", default=160, type=int)

args = parser.parse_args()

output_path = os.path.join(args.save, str(int(time.time())))
os.makedirs(output_path, exist_ok=True)

dg = DataLoaderKITTI(args.left, args.right, args.disp, args.bs, max_disp=args.maxdisp)

model = Model(args.height, args.width, args.bs, args.maxdisp, args.lr, args.cnn_3d_type)

optimizer = keras.optimizers.Adam(args.lr)
loss_func = smooth_l1_loss

model.compile(optimizer=optimizer, loss=loss_func)

print("Training Model on Data")
min_loss = np.inf
for epoch in range(1, args.epochs+1):
    print(f"\nEpoch {epoch} / {args.epochs}")
    train_loss = 0
    start_time = time.time()
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(train_size=args.train_size, is_training=True)):
        with tf.GradientTape() as tape:
            y_pred = model([imgL_crop, imgR_crop], training=True)
            disps_mask = tf.where(disp_crop_L > 0., y_pred, disp_crop_L)
            loss = model.loss(disps_mask, disp_crop_L)
            gradients = tape.gradient(loss, model.trainable_weights)
        train_loss += loss
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    avg_loss = train_loss / (160 // args.bs)
    print(f"Epoch {epoch} finished!", "Training Time:", f"{time.time()-start_time:.4f}s")
    print("Train Avg Smooth L1 Loss: %.4f" % (train_loss.numpy()))

    test_loss = 0
    start_time = time.time()
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
        y_pred = model([imgL_crop, imgR_crop], training=False)
        disps_mask = tf.where(disp_crop_L > 0., y_pred, disp_crop_L)
        loss = model.loss(disps_mask, disp_crop_L)
        test_loss += loss
    avg_loss = test_loss / (40 // args.bs)
    print("Test Avg Smooth L1 Loss: %.4f" % (test_loss.numpy()))

    if avg_loss < min_loss:
        print("Saving model!")
        model.save(os.path.join(output_path, "model"))
        min_loss = avg_loss
