# 修改 utils/logger.py
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import datetime
import os

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.5+

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建TF2版本的writer
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # 保存log_dir以便在需要时使用
        self.log_dir = log_dir
        print(f"TensorBoard logging to: {log_dir}")
        print("To view logs, run: tensorboard --logdir=" + log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert image to proper format
                if img.shape[-1] > 3:  # Check if extra channels
                    img = img[:, :, :3]  # Keep only first 3 channels
                
                # Ensure image is float and normalized [0, 1]
                if img.dtype != np.float32 and img.dtype != np.float64:
                    img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Make sure it has 3 dimensions with batch size 1
                if len(img.shape) == 2:  # grayscale
                    img = np.expand_dims(img, axis=2)
                if img.shape[2] == 1:  # grayscale with channel dim
                    img = np.repeat(img, 3, axis=2)
                
                # Add batch dimension if needed
                if len(img.shape) == 3:
                    img = np.expand_dims(img, axis=0)
                
                # Log the image
                tf.summary.image(f"{tag}/{i}", img, step=step)
            
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        # Convert values to numpy if needed
        if tf.is_tensor(values):
            values = values.numpy()
            
        # Ensure values are flattened
        values = values.flatten()
        
        # Fill in histogram
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()
            
            # Also log statistics as scalars for easier tracking
            tf.summary.scalar(f"{tag}_mean", np.mean(values), step=step)
            tf.summary.scalar(f"{tag}_std", np.std(values), step=step)
            tf.summary.scalar(f"{tag}_min", np.min(values), step=step)
            tf.summary.scalar(f"{tag}_max", np.max(values), step=step)
            self.writer.flush()
