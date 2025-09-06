#!/usr/bin/env python3
"""
3D Gaussian Splatting with KITTI data - PyTorch Training Version
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyba'))
from kitti_reader import KITTIOdometryReader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import einops
from tqdm import tqdm

SH_BASIS = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])

class Gaussian3D:
    """3D Gaussian representation"""
    def __init__(self, position: np.ndarray, scale: np.ndarray, rotation: np.ndarray, sh_coeffs: np.ndarray, opacity: np.ndarray):
        self.position:torch.Tensor = torch.from_numpy(position)  # (N, 3) - x, y, z
        self.scale:torch.Tensor = torch.from_numpy(scale)        # (N, 3) - sx, sy, sz
        self.rotation:torch.Tensor = torch.from_numpy(rotation)  # (N, 3) - rotation vector
        self.sh_coeffs:torch.Tensor = torch.from_numpy(sh_coeffs)  # (N, 3, 4) - spherical harmonics coefficients
        self.opacity:torch.Tensor = torch.from_numpy(opacity)    # (N,) - opacity

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.position = self.position.to(device)
        self.scale = self.scale.to(device)
        self.rotation = self.rotation.to(device)
        self.sh_coeffs = self.sh_coeffs.to(device)
        self.opacity = self.opacity.to(device)

        self.position.requires_grad = True
        self.scale.requires_grad = True
        self.rotation.requires_grad = True
        self.sh_coeffs.requires_grad = True
        self.opacity.requires_grad = True
        
    def get_covariance_matrix(self) -> torch.Tensor:
        """Get 3x3 covariance matrix from scale and rotation"""
        # Convert quaternion to rotation matrix
        R = self.rodrigues_batch(self.rotation)

        # Create scale matrix
        S = torch.diag_embed(self.scale)
        
        # print(f"R.shape: {R.shape}")
        # print(f"S.shape: {S.shape}")
        # Covariance = R * S * S^T * R^T

        half_cov = R * S
        cov = torch.matmul(half_cov, half_cov.transpose(1, 2))
        return cov
    
    def rodrigues_batch(self,rvecs: torch.Tensor) -> torch.Tensor:
        """
            Convert batch of rotation vectors (axis-angle) to rotation matrices.
    
            Args:
                rvecs: (N, 3) tensor of rotation vectors
    
            Returns:
                (N, 3, 3) tensor of rotation matrices
        """
        device = rvecs.device
        dtype = rvecs.dtype
    
        # Rotation angles (N,)
        theta = torch.norm(rvecs, dim=1, keepdim=True)  # (N,1)
    
        # Normalize rotation axis
        k = rvecs / (theta + 1e-8)  # avoid div by zero, (N,3)
    
        # Skew-symmetric cross-product matrices (N,3,3)
        K = torch.zeros((rvecs.shape[0], 3, 3), device=device, dtype=dtype)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]
    
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1,3,3)
    
        # Rodrigues formula: R = I + sinθ*K + (1-cosθ)*K^2
        theta = theta.view(-1, 1, 1)  # (N,1,1)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    
        return R

def init_gs(mean: np.ndarray):
    return Gaussian3D(mean, np.ones(3), np.zeros(3), np.zeros(12), np.ones(1))

def sh_basis_l1(dirs: torch.Tensor) -> torch.Tensor:
    """
    Compute 1st-order SH basis for directions.
    dirs: (N, 3) normalized directions [x, y, z]
    return: (N, 4) SH basis values
    """
    # print(f"dirs: {dirs}")
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    c0 = 0.5 / math.sqrt(math.pi)
    c1 = math.sqrt(3.0 / (4.0 * math.pi))

    Y00 = c0 * torch.ones_like(x)
    Y1m1 = c1 * y
    Y10  = c1 * z
    Y11  = c1 * x

    return torch.stack([Y00, Y1m1, Y10, Y11], dim=-1).unsqueeze(-1)  # (N, 4, 1)

def sh_coeffs_to_rgb(sh_coeffs: torch.Tensor, normal_vector: torch.Tensor) -> torch.Tensor:
    """Convert spherical harmonics coefficients to RGB"""

    base = sh_basis_l1(normal_vector)
    # print(f"sh base: {base}")
    return torch.clamp(torch.matmul(sh_coeffs, base), min=0.0, max=1.0)

def forward_gs(gs: Gaussian3D, camera_pose: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Forward pass for 3D Gaussian Splatting"""
    # Transform gaussian to camera coordinates
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    
    # Transform position
    print(f"gs.position.type: {gs.position.dtype}")
    print(f"t.type: {t.dtype}")
    print(f"R.type: {R.dtype}")
    pos_cam = (R.T @ (gs.position - t).T).T

    # Project to 2D
    x_2ds = K[0, 0] * pos_cam[:, 0] / pos_cam[:, 2] + K[0, 2]
    y_2ds = K[1, 1] * pos_cam[:, 1] / pos_cam[:, 2] + K[1, 2]

    depth = pos_cam[:, 2]
    cov_3d = gs.get_covariance_matrix()
    cov_cam = R.T @ cov_3d @ R
    # 2D covariance approximation
    N = pos_cam.shape[0]

    J = torch.zeros((N, 2, 3), device=gs.position.device)
    J[:, 0, 0] = K[0, 0] / pos_cam[:, 2]
    J[:, 0, 2] = -K[0, 0] * pos_cam[:, 0] / (pos_cam[:, 2]**2)
    J[:, 1, 1] = K[1, 1] / pos_cam[:, 2]
    J[:, 1, 2] = -K[1, 1] * pos_cam[:, 1] / (pos_cam[:, 2]**2)
    cov_2d = J @ cov_cam @ J.transpose(1, 2)

    normal_vector = gs.position - pos_cam
    normal_vector = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)

    colors = sh_coeffs_to_rgb(gs.sh_coeffs, normal_vector)
    colors = einops.rearrange(colors, 'n c 1 -> n c')
    print(f"colors.shape: {colors.shape}")
    opacity = gs.opacity
    print(f"opacity.shape: {opacity.shape}")
    
    # Initialize render buffers
    render_image = torch.zeros((height, width, 3), device=gs.position.device)
    prefix_alpha = torch.ones((height, width, 3), device=gs.position.device)
    
    # Sort by depth (back to front)
    depth_sorted_indices = torch.argsort(depth, descending=True)
    
    # Process Gaussians in batches to avoid memory explosion
    batch_size = 10  # Process fewer Gaussians at a time
    num_batches = (len(depth_sorted_indices) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Rendering batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(depth_sorted_indices))
        batch_indices = depth_sorted_indices[start_idx:end_idx]
        
        for gs_idx in batch_indices:
            x_2d = x_2ds[gs_idx]
            y_2d = y_2ds[gs_idx]
            
            # Skip if outside image bounds
            if x_2d < 0 or x_2d >= width or y_2d < 0 or y_2d >= height:
                continue
                
            # Calculate region of influence (3-sigma rule)
            cov = cov_2d[gs_idx]
            try:
                # Get eigenvalues to determine region size
                eigenvals, _ = torch.linalg.eigh(cov)
                max_std = torch.sqrt(torch.max(eigenvals))
                region_size = int(3 * max_std.item()) + 1
                
                # Define region bounds
                x_min = max(0, int(x_2d - region_size))
                x_max = min(width, int(x_2d + region_size + 1))
                y_min = max(0, int(y_2d - region_size))
                y_max = min(height, int(y_2d + region_size + 1))
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Create small grid for this region only
                mesh_x, mesh_y = np.meshgrid(
                    np.arange(x_min, x_max), 
                    np.arange(y_min, y_max)
                )
                region_grid = np.stack([mesh_x, mesh_y], axis=-1)
                region_grid = torch.from_numpy(region_grid).to(gs.position.device)
                
                coordinate = torch.tensor([y_2d, x_2d], device=gs.position.device)
                diff = region_grid - coordinate
                
                cov_inv = torch.linalg.inv(cov)
                first = torch.matmul(diff, cov_inv)
                second = einops.einsum(first, diff, "H W d, H W d -> H W")
                gaussian_value = torch.exp(-0.5 * second)
                
                # Apply color and opacity
                color = colors[gs_idx, :]  # Shape: (3,)
                o = opacity[gs_idx]
                
                # Efficient broadcasting for small region
                alpha = o * gaussian_value.unsqueeze(-1)  # Shape: (H, W, 1)
                
                # Update only the affected region
                render_image[y_min:y_max, x_min:x_max] += color.unsqueeze(0).unsqueeze(0) * alpha * prefix_alpha[y_min:y_max, x_min:x_max]
                prefix_alpha[y_min:y_max, x_min:x_max] *= (1 - alpha)
                
            except torch.linalg.LinAlgError:
                # Skip singular covariance matrices
                continue
            except Exception as e:
                print(f"Error processing Gaussian {gs_idx}: {e}")
                continue
    
    return render_image

def forward_gs_cpu(gs: Gaussian3D, camera_pose: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """CPU-based forward pass for 3D Gaussian Splatting to avoid memory issues"""
    # Move everything to CPU
    device = torch.device('cpu')
    
    # Transform gaussian to camera coordinates
    R = camera_pose[:3, :3].to(device)
    t = camera_pose[:3, 3].to(device)
    K_cpu = K.to(device)
    
    # Move Gaussian data to CPU
    position_cpu = gs.position.to(device)
    scale_cpu = gs.scale.to(device)
    rotation_cpu = gs.rotation.to(device)
    sh_coeffs_cpu = gs.sh_coeffs.to(device)
    opacity_cpu = gs.opacity.to(device)
    
    # Transform position
    pos_cam = (R.T @ (position_cpu - t).T).T

    mean2D = (K_cpu @ pos_cam.T).T
    mean2D = mean2D[:, :2] / mean2D[:, 2:3]
    # print(f"mean2D.shape: {mean2D.shape}")
    # print(f"mean2D: {mean2D}")

    depth = pos_cam[:, 2]
    # print(f"depth: {depth}")
    normal_vector = position_cpu - t
    normal_vector = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)
    # print(f"normal_vector: {normal_vector}")
    
    # Get covariance matrices (simplified for CPU)
    cov_3d = gs.get_covariance_matrix()
    # print(f"cov_3d : {cov_3d}")
    cov_cam = R.T @ cov_3d @ R
    # print(f"cov_cam.shape: {cov_cam.shape}")
    # 2D covariance approximation
    N = pos_cam.shape[0]
    J = torch.zeros((N, 2, 3), device=device)
    J[:, 0, 0] = K_cpu[0, 0] / pos_cam[:, 2]
    J[:, 0, 2] = -K_cpu[0, 0] * pos_cam[:, 0] / (pos_cam[:, 2]**2)
    J[:, 1, 1] = K_cpu[1, 1] / pos_cam[:, 2]
    J[:, 1, 2] = -K_cpu[1, 1] * pos_cam[:, 1] / (pos_cam[:, 2]**2)
    cov_2d = J @ cov_cam @ J.transpose(1, 2)

    # print(f"cov_2d: {cov_2d}")

    # Simple color calculation (no spherical harmonics for now)
    colors = sh_coeffs_to_rgb(sh_coeffs_cpu, normal_vector)
    # print(f"colors: {colors}")
    
    # Initialize render buffers
    render_image = torch.zeros((height, width, 3), device=device)
    
    # Sort by depth (back to front)
    depth_sorted_indices = torch.argsort(depth, descending=True)
    
    # Simplified approach: additive blending without complex alpha compositing
    # This is gradient-friendly but doesn't implement proper depth ordering
    
    # Process a subset of Gaussians to avoid memory issues
    # max_gaussians = min(128, len(depth_sorted_indices))  # Limit to 100 gaussians

    x, y = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    mesh = torch.stack([y, x], dim=-1)
    mesh = mesh.to(device)

    prefix_alpha = torch.ones((height, width, 3), device=device)
    for i, gs_idx in enumerate(tqdm(depth_sorted_indices, desc="Rendering Gaussians")):
        diff = mesh - mean2D[gs_idx, :]
        cov = torch.linalg.inv(cov_2d[gs_idx])
        first = einops.einsum(diff, cov, "H W d, d d -> H W d")
        second = einops.einsum(first, diff, "H W d, H W d -> H W")
        gaussian_value = torch.exp(-0.5 * second)
        alpha = einops.repeat(opacity_cpu[gs_idx] * gaussian_value, "H W -> H W c", c=3)
        color = colors[gs_idx, :, 0]
        render_image = render_image + alpha * color * prefix_alpha
        prefix_alpha = prefix_alpha * (1 - alpha)
    # print(f"render_image[0, 0, 0]: {render_image[0, 0, 0]}")
    return render_image


def initialize_gaussians_from_image(image: np.ndarray,  K: np.ndarray, depth_estimate: float = 10.0) -> List[Gaussian3D]:
    """Initialize 3D gaussians from image pixels"""
    height, width = image.shape[:2]
    gaussians = []

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # Sample pixels (every 4th pixel for efficiency)
    step = 32
    means = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Convert pixel to 3D position (simple depth assumption)
            z_3d = depth_estimate + np.random.normal(0, 1.0)  # Add some depth variation
            x_3d = (x - cx) * z_3d / fx  # Simple perspective
            y_3d = (y - cy) * z_3d / fy
            means.append(np.array([x_3d, y_3d, z_3d]))
            
    means = np.array(means, dtype=np.float32)
    scales = np.ones((means.shape[0], 3), dtype=np.float32)
    rotations = np.zeros((means.shape[0], 3), dtype=np.float32)
    sh_coeffs = np.zeros((means.shape[0], 3, 4), dtype=np.float32)
    opacities = np.ones(means.shape[0], dtype=np.float32) * 0.5
    
    gaussians = Gaussian3D(means, scales, rotations, sh_coeffs, opacities)
    return gaussians

def visualize_training_results(renderer, gaussians, dataset, device, epoch):
    """Visualize training results"""
    with torch.no_grad():
        # Get first frame
        first_batch = dataset[0]
        target_image = first_batch['image'].to(device)
        camera_pose = first_batch['pose'].to(device)
        
        # Render
        rendered_image = renderer(gaussians, camera_pose)
        
        # Convert to numpy for visualization
        target_np = target_image.cpu().numpy()
        rendered_np = rendered_image.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(target_np)
        axes[0].set_title('Target Image')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(rendered_np, 0, 1))
        axes[1].set_title(f'Rendered Image (Epoch {epoch})')
        axes[1].axis('off')
        
        diff = np.abs(target_np - rendered_np)
        axes[2].imshow(diff)
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'training_result_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def create_3dgs_from_kitti():
    """Create 3D Gaussian Splatting from KITTI data"""
    
    # Load KITTI data
    reader = KITTIOdometryReader()
    sequence_id = reader.sequences[5]  # Using sequence 5 as in your code
    
    # Load first image and calibration
    left_img = reader.load_image(sequence_id, 0, 'left')
    calib = reader.load_calibration(sequence_id)
    poses = reader.load_poses(sequence_id)
    
    # Extract camera parameters from calibration
    P2 = calib['P2']  # Left camera projection matrix
    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    
    print(f"Camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"Image shape: {left_img.shape}")
    
    # Initialize 3D Gaussian Splatting
    height, width = left_img.shape[:2]
    renderer = GaussianSplatting3D(width, height, fx, fy, cx, cy)
    
    # Initialize gaussians from first image
    print("Initializing 3D gaussians from image...")
    gaussians = initialize_gaussians_from_image(left_img)
    print(f"Created {len(gaussians)} gaussians")
    
    # Add gaussians to renderer
    for gaussian in gaussians:
        renderer.add_gaussian(gaussian)
    
    # Render from first camera pose
    first_pose = poses[0]
    print("Rendering from first camera pose...")
    rendered_image = renderer.render(first_pose)
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original KITTI Image')
    axes[0].axis('off')
    
    # Rendered image
    axes[1].imshow(np.clip(rendered_image, 0, 1))
    axes[1].set_title('3D Gaussian Splatting Render')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - rendered_image)
    axes[2].imshow(diff)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Render from different viewpoints
    print("Rendering from different viewpoints...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, frame_idx in enumerate([0, 10, 20, 30, 40, 50]):
        if frame_idx < len(poses):
            pose = poses[frame_idx]
            rendered = renderer.render(pose)
            
            row = i // 3
            col = i % 3
            axes[row, col].imshow(np.clip(rendered, 0, 1))
            axes[row, col].set_title(f'Frame {frame_idx}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return renderer, gaussians, poses

if __name__ == "__main__":

    left_img = cv2.imread("000000.png")
    height, width = left_img.shape[:2]
    camera_pose = np.eye(4)
    P0 = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00],
                   [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00],
                   [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
    K = P0[:3, :3]

    gaussians = initialize_gaussians_from_image(left_img, K)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_tensor = torch.from_numpy(camera_pose).float().to(device)
    K_tensor = torch.from_numpy(K).float().to(device)

    optimizer = optim.Adam([gaussians.position, gaussians.scale, gaussians.rotation, gaussians.sh_coeffs, gaussians.opacity], lr=1e-2)
    criterion = nn.L1Loss()
    print("Starting rendering iterations...")
    # print(f"gaussians.position: {gaussians.position}")
    # print(f"gaussians.scale: {gaussians.scale}")
    # print(f"gaussians.rotation: {gaussians.rotation}")
    # print(f"gaussians.sh_coeffs: {gaussians.sh_coeffs}")
    # print(f"gaussians.opacity: {gaussians.opacity}")
    target_image = torch.from_numpy(left_img).float().to(device) / 255.0
    max_iterations = 1024
    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")
        optimizer.zero_grad()
        rendered_image = forward_gs_cpu(gaussians, pose_tensor, K_tensor, width, height)
        print(f"Rendered image shape: {rendered_image.shape}")
        print(f"Rendered image min/max: {rendered_image.min():.3f}/{rendered_image.max():.3f}")
        loss = criterion(rendered_image, target_image)
        print(f"Loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()

        # write rendered image to file
        rendered_image_np = rendered_image.detach().cpu().numpy()
        rendered_image_np = (rendered_image_np * 255).astype(np.uint8)
        os.makedirs("rendered_image", exist_ok=True)
        cv2.imwrite(f"rendered_image/{i}.png", rendered_image_np)
    print("Final rendering...")

    gaussians.position = gaussians.position.detach()
    gaussians.scale = gaussians.scale.detach()
    gaussians.rotation = gaussians.rotation.detach()
    gaussians.sh_coeffs = gaussians.sh_coeffs.detach()
    gaussians.opacity = gaussians.opacity.detach()
    # print(f"gaussians.position: {gaussians.position}")
    # print(f"gaussians.scale: {gaussians.scale}")
    # print(f"gaussians.rotation: {gaussians.rotation}")
    # print(f"gaussians.sh_coeffs: {gaussians.sh_coeffs}")
    # print(f"gaussians.opacity: {gaussians.opacity}")

    rendered_image = forward_gs_cpu(gaussians, pose_tensor, K_tensor, width, height)
    print(f"Final rendered image shape: {rendered_image.shape}")
    print(f"Final rendered image min/max: {rendered_image.min():.3f}/{rendered_image.max():.3f}")
    # Convert to numpy and display
    rendered_np = rendered_image.detach().cpu().numpy() 
    print(f"Numpy image shape: {rendered_np.shape}")
    print(f"Numpy image min/max: {rendered_np.min():.3f}/{rendered_np.max():.3f}")
        
    plt.figure(figsize=(12, 8))
    plt.imshow(rendered_np)
    plt.title('3D Gaussian Splatting Render')
    plt.colorbar()
    plt.show()
    print("Plot displayed successfully!")
        