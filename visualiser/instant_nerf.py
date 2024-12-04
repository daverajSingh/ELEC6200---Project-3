import math
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

def load_data(path):
    data = np.load(path)
    images = torch.from_numpy(data['images'])
    poses = torch.from_numpy(data['poses'])
    focal = float(data['focal'])
    H, W = images.shape[1:3]
    
    testimg, testpose = images[101], poses[101]
    
    return images, poses, focal, H, W, testimg, testpose

def scale_images(images, scale_factor=2):
    n = images.shape[0]
    h = images.shape[1] // scale_factor
    w = images.shape[2] // scale_factor
    
    # Average pooling by reshaping and taking mean
    if len(images.shape) == 4:  # With channels
        return images.reshape(n, h, scale_factor, w, scale_factor, -1).mean(axis=(2, 4))
    else:  # No channels
        return images.reshape(n, h, scale_factor, w, scale_factor).mean(axis=(2, 4))

@torch.no_grad()
def get_output_for_img_iter(nerf_model, hn, hf, nb_bins, testpose, H, W, focal, batch_size, N = 16, flag=None, pbar=None):
    rays_o, rays_d = get_rays(H, W, focal, testpose)
    rays_o = rays_o.reshape((-1, 3)).to(device)
    rays_d = rays_d.reshape((-1, 3)).to(device)

    rays_o_split = [rays_o[i::N] for i in range(N)]
    rays_d_split = [rays_d[i::N] for i in range(N)]

    img_stack = np.array([])
    seg_img_stack = np.array([])

    if pbar is None:
        pbar = tqdm(total=N, desc="Processing")
    
    pbar.reset()

    for ind, (rays_o, rays_d) in enumerate(zip(rays_o_split, rays_d_split)):
        num_rays = rays_o.shape[0]
        pred_px_values = []
        pred_seg_values = []
        for i in range(0, num_rays, batch_size):
            if flag is not None and flag.is_set():
                return None
            batch_rays_o = rays_o[i:i+batch_size]
            batch_rays_d = rays_d[i:i+batch_size]
            batch_pred_px_values, batch_pred_seg_values = render_rays(nerf_model, batch_rays_o, batch_rays_d, hn=hn, hf=hf, nb_bins=nb_bins)
            pred_px_values.append(batch_pred_px_values)
            pred_seg_values.append(batch_pred_seg_values)

        pred_px_values = torch.cat(pred_px_values, dim=0).cpu().numpy()
        pred_seg_values = torch.cat(pred_seg_values, dim=0).argmax(dim=1).cpu().numpy()

        if len(img_stack) == 0:
            img_stack = np.array([pred_px_values])
            seg_img_stack = np.array([pred_seg_values])
            img = pred_px_values
            seg_img = pred_seg_values
        else:
            img_stack = np.vstack((img_stack, pred_px_values[np.newaxis, :, :]))
            seg_img_stack = np.vstack((seg_img_stack, pred_seg_values[np.newaxis, :]))
            img = img_stack.reshape((-1,3),order='F')
            seg_img = seg_img_stack.reshape((-1),order='F')
        
        img = np.repeat(img, (H*W) // len(img), axis=0)[:H*W]
        seg_img = np.repeat(seg_img, (H*W) // len(seg_img), axis=0)[:H*W]
        pbar.update(1)

        if img.shape[0] == H*W:
            img = img.reshape(H, W, 3).clip(0, 1)*1.
            seg_img = seg_img.reshape(H, W)
            # print(f"{((ind+1) / N) * 100} %")
            yield img, seg_img

    # print("Image fully rendered")
    yield img, seg_img
    return

@torch.no_grad()
def get_output_for_img(nerf_model, hn, hf, nb_bins, testpose, H, W, focal, batch_size):
    rays_o, rays_d = get_rays(H, W, focal, testpose)
    rays_o = rays_o.reshape((-1, 3)).to(device)
    rays_d = rays_d.reshape((-1, 3)).to(device)

    num_rays = rays_o.shape[0]
    pred_px_values = []
    pred_seg_values = []

    for i in range(0, num_rays, batch_size):
        batch_rays_o = rays_o[i:i+batch_size]
        batch_rays_d = rays_d[i:i+batch_size]
        batch_pred_px_values, batch_pred_seg_values = render_rays(nerf_model, batch_rays_o, batch_rays_d, hn=hn, hf=hf, nb_bins=nb_bins)
        pred_px_values.append(batch_pred_px_values)
        pred_seg_values.append(batch_pred_seg_values)

    pred_px_values = torch.cat(pred_px_values, dim=0)
    pred_seg_values = torch.cat(pred_seg_values, dim=0).argmax(dim=1)
    
    print_memory_usage()
    img = pred_px_values.data.cpu().reshape(H, W, 3)
    seg = pred_seg_values.data.cpu().reshape(H, W)
    img = (img.clip(0, 1)*1.)
    # seg = (seg.clip(0, 1)*1.)

    return img, seg

class NGP(torch.nn.Module):
    def __init__(self, T, Nl, L, device, aabb_scale, F=2, num_of_labels=10):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.L = L  # For encoding directions
        self.aabb_scale = aabb_scale
        self.num_of_labels = num_of_labels
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T, 2), device=device) * 2 - 1) * 1e-4) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861
        self.density_MLP = nn.Sequential(nn.Linear(self.F * len(Nl), 64),
                                         nn.ReLU(), nn.Linear(64, 16)).to(device)
        self.color_MLP = nn.Sequential(nn.Linear(27 + 16, 64), nn.ReLU(),
                                       nn.Linear(64, 64), nn.ReLU(),
                                       nn.Linear(64, 3), nn.Sigmoid()).to(device)

        self.seg_MLP = nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(),
                                nn.Linear(64, num_of_labels), nn.Softmax()).to(device)

    def positional_encoding(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):

        x /= self.aabb_scale
        mask = (x[:, 0].abs() < .5) & (x[:, 1].abs() < .5) & (x[:, 2].abs() < .5)
        x += 0.5  # x in [0, 1]^3

        color = torch.zeros((x.shape[0], 3), device=x.device)
        seg = torch.zeros((x.shape[0], self.num_of_labels), device=x.device)
        log_sigma = torch.zeros((x.shape[0]), device=x.device) - 100000
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
        for i, N in enumerate(self.Nl):
            # Computing vertices, use nn.functional.grid_sample convention
            floor = torch.floor(x[mask] * N)
            ceil = torch.ceil(x[mask] * N)
            vertices = torch.zeros((x[mask].shape[0], 8, 3), dtype=torch.int64, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat((ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 7] = ceil

            # hashing
            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)

            # Lookup
            looked_up = self.lookup_tables[str(i)][h_x].transpose(-1, -2)
            volume = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))
            features[:, i*2:(i+1)*2] = torch.nn.functional.grid_sample(
                volume,
                ((x[mask] * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                ).squeeze(-1).squeeze(-1).squeeze(-1)

        xi = self.positional_encoding(d[mask])
        h = self.density_MLP(features)
        log_sigma[mask] = h[:, 0]
        color_output = self.color_MLP(torch.cat((h, xi), dim=1))
        color[mask] = color_output
        seg[mask] = self.seg_MLP(h)
        
        return color, torch.exp(log_sigma), seg

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),
                         torch.arange(H, dtype=torch.float32, device=device),
                         indexing='xy')
    
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor(
        [1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    colors, sigma, segs = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors.reshape(x.shape)).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background

    s = (weights * segs.reshape(x.shape[0], x.shape[1], -1)).sum(dim=1)
    return c + 1 - weight_sum.unsqueeze(-1), s + 1 - weight_sum.unsqueeze(-1)

def print_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Total memory: {t}, Reserved: {r}, Free: {f}")

def get_balanced_indices(targets, seed=None):
    """
    Returns indices for balanced sampling from each class in a PyTorch tensor.
    Arguments:
        targets: A 1D PyTorch tensor of shape (batch_size,) containing class labels.
        seed: Random seed for reproducibility (default: None).
    Returns:
        indices: A 1D PyTorch tensor of balanced indices.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Get unique classes and their indices
    classes = torch.unique(targets)
    
    # Group indices by class
    class_indices = {cls.item(): torch.where(targets == cls)[0] for cls in classes}
    
    # Determine the median class size
    class_sizes = torch.tensor([len(indices) for indices in class_indices.values()])
    median_class_size = int(torch.median(class_sizes).item())
    
    # Sample indices up to the median size
    balanced_indices = []
    for cls in classes:
        indices = class_indices[cls.item()]
        sampled_indices = indices[torch.randperm(len(indices))[:median_class_size]]
        balanced_indices.append(sampled_indices)
    
    # Combine and shuffle the balanced indices
    balanced_indices = torch.cat(balanced_indices)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
    
    return balanced_indices

def train(nerf_model, optimizer, data_loader, testimg, test_img_seg, testpose, device='cpu', hn=0, hf=1, nb_epochs=10,
          nb_bins=192, H=400, W=400, focal=0, i_plot=1, batch_size=None):
    loss_per_iter = []
    psnrs = []
    iternums = []
    t = time.time()
    print("Starting training")    
    for epoch in range(nb_epochs):
        losses = []
        i = 0
        cur_time = time.time()
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:9].to(device)
            gt_seg_values = batch[:, 9:].reshape(-1).to(device).to(torch.int64)

            pred_px_values, pred_seg_values = render_rays(nerf_model, ray_origins, ray_directions, 
                                         hn=hn, hf=hf, nb_bins=nb_bins)

            loss = ((gt_px_values - pred_px_values) ** 2).mean()
            losses.append(loss)

            seg_inds = get_balanced_indices(gt_seg_values)
            loss += nn.CrossEntropyLoss()(pred_seg_values[seg_inds], gt_seg_values[seg_inds]).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(i, time.time() - t)
            i += 1
            # if i > 10:
            #     break

        avg_loss = sum(losses)/len(losses)
        print(epoch, avg_loss)
        loss_per_iter.append(avg_loss.item())

        if epoch % i_plot == 0:
            print(f'Iteration {epoch}: {(time.time() - t) / i_plot:.2f} secs per iter')
            t = time.time()
            
            # Render the holdout view for logging
            with torch.no_grad():
                img, seg = get_output_for_img(nerf_model, hn=hn, hf=hf, nb_bins=nb_bins, testpose=testpose, H=H, W=W, focal=focal, batch_size=batch_size)
                loss = torch.mean((img - testimg) ** 2)
                psnr = -10. * torch.log(loss) / torch.log(torch.tensor(10.))
                
                psnrs.append(psnr.item())
                iternums.append(epoch)

                fig, axs = plt.subplots(2, 3, figsize=(10, 10))
                axs[0, 0].imshow(seg.cpu())
                axs[0, 0].set_title('Segmentation')

                axs[1, 0].imshow(img.cpu())
                axs[1, 0].set_title('Rendered Image')

                axs[0, 1].imshow(test_img_seg.cpu())
                axs[0, 1].set_title('Ground Truth Segmentation')

                axs[1, 1].imshow(testimg.cpu())
                axs[1, 1].set_title('Ground Truth Image')

                axs[0, 2].plot(iternums, psnrs)
                axs[0, 2].set_ylabel('PSNR')
                axs[0, 2].set_xlabel('Iteration')
                axs[0, 2].set_title('PSNR')
                axs[1, 2].plot([j for j in range(epoch+1)], loss_per_iter)
                axs[1, 2].set_title('Loss')
                axs[1, 2].set_ylabel('Loss')
                axs[1, 2].set_xlabel('Iteration')
            

                plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
                plt.savefig(os.path.join(PLOT_PATH, f'loss_plot_{epoch}.png'), bbox_inches='tight', dpi=300)
                torch.save(nerf_model.module.state_dict(), os.path.join(MODEL_PATH, f'model_state_dict_{epoch}.pth'))
                plt.close()

def load_model(model_name, num_of_labels):
    nerf_model = NGP(T, Nl, 4, device, 16, num_of_labels=num_of_labels) # CHANGE AABB TO 16 from 3

    checkpoint = torch.load(os.path.join("", model_name), map_location=torch.device(device))

    nerf_model.load_state_dict(checkpoint)

    nerf_model.eval()
    return nerf_model

def get_output(nerf_model, pose, H, W, focal):
    img, seg = get_output_for_img(nerf_model, hn=HN, hf=HF, nb_bins=NB_BINS, testpose=pose, H=H, W=W, focal=focal, batch_size=batch_size)
    return img, seg

def main(data_path):
    SCALE_FACTOR = 4

    loaded = np.load(data_path)
    images = torch.from_numpy(scale_images(loaded['images_train'], scale_factor=SCALE_FACTOR))[0::10]
    seg_images = torch.from_numpy(scale_images(loaded['seg_images'], scale_factor=SCALE_FACTOR))[0::10]
    poses = torch.from_numpy(loaded['poses_train'])[0::10]

    W = loaded['W'].item() // SCALE_FACTOR
    H = loaded['H'].item() // SCALE_FACTOR
    focal = loaded['focal'].item()
    print("Running nerf, number of training images: ", len(images))
    training_data = []
    for img, seg_img, pose in zip(images, seg_images, poses):
        rays_o, rays_d = get_rays(H, W, focal, pose)
        training_data.append(torch.cat([rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.reshape(-1, 3), seg_img.reshape(-1, 1)], dim=1))

    training_dataset = torch.cat(training_data, dim=0)
    testimg, test_img_seg, testpose = images[0], seg_images[0], poses[0]

    PLOT_PATH = 'plots'
    os.makedirs(PLOT_PATH, exist_ok=True)
    MODEL_PATH = 'models'
    os.makedirs(MODEL_PATH, exist_ok=True)


    model = NGP(T, Nl, 4, device, 16, num_of_labels=4)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUS: ", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)
    model_optimizer = torch.optim.Adam(
        [{"params": model.module.lookup_tables.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
         {"params": model.module.density_MLP.parameters(), "lr": 1e-2,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
         {"params": model.module.color_MLP.parameters(), "lr": 1e-2,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
         {"params": model.module.seg_MLP.parameters(), "lr": 1e-2,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6}])
    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    train(model, model_optimizer, data_loader, testimg, test_img_seg, testpose, nb_epochs=100, device=device,
          hn=HN, hf=HF, nb_bins=NB_BINS, H=H, W=W, focal=focal, batch_size=batch_size)

PLOT_PATH = 'plots'
os.makedirs(PLOT_PATH, exist_ok=True)
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU is not available. Running on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"Instant ngp specific vars"
L = 16
F = 2
T = 2**19
N_min = 16
N_max = 2048
batch_size=2**12
b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
HN = 2
HF = 6
NB_BINS= 192

if __name__ == "__main__":
    main("nerf_formated_data.npz")
    # model = load_model('model_state_dict_18.pth')
