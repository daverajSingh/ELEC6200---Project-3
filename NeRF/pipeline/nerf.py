import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
# import format_data
    
def load_data(path):
    data = np.load(path)
    images = torch.from_numpy(data['images'])
    poses = torch.from_numpy(data['poses'])
    focal = float(data['focal'])
    H, W = images.shape[1:3]
    
    # Split into train and test
    testimg, testpose = images[101], poses[101]
    # images = images[:100,...,:3]
    # poses = poses[:100]
    
    return images, poses, focal, H, W, testimg, testpose

def posenc(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, -1)

class NeRF(nn.Module):
    def __init__(self, D=12, W=256, input_ch=39):  # input_ch = 3 + 3*2*L_embed when L_embed=6
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        
        # Create layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_ch, W))
        
        for i in range(D-1):
            if i % 4 == 0 and i > 0:
                self.layers.append(nn.Linear(W + input_ch, W))
            else:
                self.layers.append(nn.Linear(W, W))
                
        self.output_layer = nn.Linear(W, 4)
        
    def forward(self, x):
        inputs = x
        x = inputs
        
        for i in range(self.D):
            x = self.layers[i](x)
            x = torch.relu(x)
            if i % 4 == 0 and i > 0:
                x = torch.cat([x, inputs], -1)
                
        outputs = self.output_layer(x)
        return outputs

def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),
                         torch.arange(H, dtype=torch.float32, device=device),
                         indexing='xy')
    
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    # Compute 3D query points
    t_vals = torch.linspace(near, far, N_samples, device=rays_o.device)
    
    # Expand t_vals to match the shape of rays_o
    z_vals = t_vals.expand(rays_o.shape[:-1] + (N_samples,)).clone()

    if rand:
        z_vals += torch.rand(list(rays_o.shape[:-1]) + [N_samples], device=rays_o.device) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    # Flatten query points and run network
    pts_flat = pts.reshape(-1, 3)
    pts_flat = posenc(pts_flat)
    raw = network_fn(pts_flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])
    
    # Compute opacities and colors
    sigma_a = torch.relu(raw[...,3])
    rgb = torch.sigmoid(raw[...,:3])
    
    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                      torch.ones_like(z_vals[...,:1]) * 1e10], -1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]),
                                             1.-alpha + 1e-10], -1), -1)[...,:-1]
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    return rgb_map, depth_map, acc_map

def train_nerf(images, model, poses, H, W, focal, testimg, testpose, N_samples=16, N_iters=1000, freq=200, device='cuda'):
    images = images.to(device)
    poses = poses.to(device)
    testimg = testimg.to(device)
    testpose = testpose.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    i_plot = freq

    psnrs = []
    loss_per_iter = []

    iternums = []
    t = time.time()
    
    for i in range(N_iters + 1):
        # img_i = np.random.randint(images.shape[0])
        losses = []
        for img_i in range(images.shape[0]):
            target = images[img_i]
            pose = poses[img_i]
            rays_o, rays_d = get_rays(H, W, focal, pose)
            
            optimizer.zero_grad()
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., 
                                        N_samples=N_samples, rand=True)
            loss = torch.mean((rgb - target) ** 2)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        
        avg_loss = sum(losses)/len(losses)
        print(i, avg_loss)
        loss_per_iter.append(avg_loss.item())
        
        if i % i_plot == 0 and i > 0:
            print(f'Iteration {i}: {(time.time() - t) / i_plot:.2f} secs per iter')
            t = time.time()
            
            # Render the holdout view for logging
            with torch.no_grad():
                rays_o, rays_d = get_rays(H, W, focal, testpose)
                rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., 
                                            N_samples=N_samples)
                loss = torch.mean((rgb - testimg) ** 2)
                psnr = -10. * torch.log(loss) / torch.log(torch.tensor(10.))
                
                psnrs.append(psnr.item())
                iternums.append(i)

                plt.figure(figsize=(10,10))
                plt.subplot(121)
                plt.imshow(rgb.cpu())
                plt.title(f'Iteration: {i}')
                ax2 = plt.subplot(122)
                ax2.plot(iternums, psnrs)
                ax2.set_ylabel('PSNR')
                ax2.set_xlabel('Iteration')
                ax2.set_title('PSNR')
                ax3 = plt.subplot(223)
                ax3.plot([j for j in range(i+1)], loss_per_iter)
                ax3.set_title('Loss')
                ax3.set_ylabel('Loss')
                ax3.set_xlabel('Iteration')

                plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
                plt.savefig(os.path.join(PLOT_PATH, f'loss_plot_{i}.png'), bbox_inches='tight', dpi=300)
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'model_state_dict_{i}.pth'))
                plt.close()
                # plt.show()
    
    return model, psnrs

def visualize_images(images, images_per_row=10):
    # Total number of images and images per row
    n_images = len(images)
    images_per_column = int(np.ceil(n_images / images_per_row))

    # Set figure size based on grid dimensions
    plt.figure(figsize=(20, 20))  # Adjust as needed for better resolution

    for idx, image in enumerate(images[:100]):  # Limit to 1000 images
        plt.subplot(images_per_column, images_per_row, idx + 1)
        plt.imshow(image, cmap='gray')  # Use 'gray' for grayscale images, remove for RGB
        plt.axis('off')  # Hide axes for clean display

    plt.tight_layout()
    plt.show()


def main(data_path, visualise=False):
    loaded = np.load(data_path)
    images = torch.from_numpy(loaded['images_train'])
    poses = torch.from_numpy(loaded['poses_train'])
    W = loaded['W'].item()
    H = loaded['H'].item()
    focal = loaded['focal'].item()
    testimg, testpose = images[0], poses[0]

    if visualise:
        visualise(images)

    print(images.shape, poses.shape, focal, H, W)

    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU is not available. Running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRF().to(device)
    model, psnrs = train_nerf(images, model, poses, H, W, focal, testimg, testpose, device=device, freq=50)

PLOT_PATH = 'plots'
os.makedirs(PLOT_PATH, exist_ok=True)
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

if __name__ == "__main__":
    main("nerf_formated_data.npz")