import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

def load_data(path: str):
    data = np.load('tiny_nerf_data.npz')
    images = data['images'] # (batch_size, X, Y, channels)
    poses = data['poses'] # (106, 4, 4), 4x4 matrix transformation matrix
    focal = data['focal'] # Focal point of the camera
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    testimg, testpose = images[101], poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]

    return images, poses, focal, H, W

class Neural_Radiance_Fields():
    def __init__(self, image_h, image_w, focal, L_embed=6, embed_fn=None):
        self.L_embed = L_embed
        self.image_h = image_h
        self.image_w = image_w
        self.focal = focal

        if embed_fn is None:
           embed_fn = self.posnec
        self.embed_fn = embed_fn
        self.model = self.init_model()
        self.optimizer = tf.keras.optimizers.Adam(5e-4) # Adam optimiser used for training

    def posnec(self, x):
        """
        Applies position embedding on x breaking down the position into sin and cosing co-effecients.
        """
        rets = [x]
        for i in range(self.L_embed):
            for fn in [tf.sin, tf.cos]:
                rets.append(fn(2.**i * x))
        return tf.concat(rets, -1)
    
    def init_model(self, D=8, W=256):
        """
        Creates model which stores the scene representation
        """
        relu = tf.keras.layers.ReLU()    
        dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act) # FC layer with ReLU activation.

        inputs = tf.keras.Input(shape=(3 + 3*2*self.L_embed)) # Input is rotation + (r,g,b)*(sin,cos)*(number of positional layers)
        outputs = inputs
        for i in range(D): # 8 FC layers, with skip connections every 4 layers.
            outputs = dense()(outputs) 
            if i%4==0 and i>0: 
                outputs = tf.concat([outputs, inputs], -1)
        outputs = dense(4, act=None)(outputs) # Final FC layer to get outputs. (rgb, density)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model

    def get_rays(self, H, W, focal, c2w):
        """
        Transforms each pixel in the image to world-coordiantes by using a prespective projection matrix (c2w) and the focal angle.
        Returns the ray_origins in world-coordinates and ray direction.
        """
        i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
        dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
        rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
        return rays_o, rays_d

    def render_rays(self, network_fn, rays_o, rays_d, near, far, N_samples, rand=False):

        def batchify(fn, chunk=1024*32):
            return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        
        # Compute 3D query points
        z_vals = tf.linspace(near, far, N_samples) 
        if rand:
            z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        
        # Run network
        pts_flat = tf.reshape(pts, [-1,3])
        pts_flat = self.embed_fn(pts_flat)
        raw = batchify(network_fn)(pts_flat)
        raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])
        
        # Compute opacities and colors
        sigma_a = tf.nn.relu(raw[...,3])
        rgb = tf.math.sigmoid(raw[...,:3]) 
        
        # Do volume rendering
        dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1) 
        alpha = 1.-tf.exp(-sigma_a * dists)  
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        
        rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2) 
        depth_map = tf.reduce_sum(weights * z_vals, -1) 
        acc_map = tf.reduce_sum(weights, -1)

        return rgb_map, depth_map, acc_map

    def train_on_one_sample(self, target, pose, N_samples):
        rays_o, rays_d = self.get_rays(self.image_h, self.image_w, self.focal, pose)

        with tf.GradientTape() as tape:
            rgb, depth, acc = self.render_rays(self.model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - target))
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    def get_result_for_image(self, pose):
        rays_o, rays_d = self.get_rays(self.image_h, self.image_w, self.focal, pose)
        rgb, depth, acc = self.render_rays(self.model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        return rgb


images, poses, focal, H, W = load_data('tiny_nerf_data.npz')
model = Neural_Radiance_Fields(image_h=H, image_w=W, focal=focal)

N_samples = 16
N_iters = 1000
psnrs = []
iternums = []
i_plot = 200

import time
t = time.time()
for i in range(N_iters+1):
    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    model.train_on_one_sample(target, pose, N_samples)

    if i%i_plot==0:
        image = model.get_result_for_image(poses[i])
        plt.imshow(image)
        plt.show()
