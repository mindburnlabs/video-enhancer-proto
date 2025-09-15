"""
Noise Scheduler for diffusion-based video restoration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

class NoiseScheduler:
    """
    Noise scheduler for diffusion models with various scheduling strategies.
    """
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 prediction_type: str = "epsilon"):
        
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        
        # Generate beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # Scaled linear schedule from DDPM
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "cosine":
            # Cosine schedule
            s = 0.008
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Derived quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, 
                  original_samples: torch.Tensor, 
                  noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to original samples according to noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Noise to add
            timesteps: Timesteps for each sample
            
        Returns:
            Noisy samples
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Expand dimensions to match original_samples
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_velocity(self, 
                     sample: torch.Tensor, 
                     noise: torch.Tensor, 
                     timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get velocity parameterization for v-prediction.
        
        Args:
            sample: Clean sample
            noise: Added noise
            timesteps: Timesteps
            
        Returns:
            Velocity
        """
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Expand dimensions
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def step(self, 
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Predict the sample at the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in diffusion chain
            sample: Current instance of sample being created by diffusion process
            generator: Random number generator
            
        Returns:
            Previous sample
        """
        t = timestep
        
        # Get parameters
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev**(0.5) * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t]**(0.5) * beta_prod_t_prev / beta_prod_t
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise if not the final step
        if t > 0:
            device = sample.device
            variance = self.posterior_variance[t]
            if generator is not None:
                noise = torch.randn(sample.shape, generator=generator, device=device, dtype=sample.dtype)
            else:
                noise = torch.randn(sample.shape, device=device, dtype=sample.dtype)
            pred_prev_sample = pred_prev_sample + (variance**(0.5)) * noise
        
        return pred_prev_sample
    
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Scale the input sample for compatibility with the scheduler.
        
        Args:
            sample: Input sample
            timestep: Current timestep (unused for this scheduler)
            
        Returns:
            Scaled sample
        """
        return sample
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of inference steps
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        
        # Create timestep schedule
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def previous_timestep(self, timestep: int) -> int:
        """Get the previous timestep."""
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is not set")
        
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t