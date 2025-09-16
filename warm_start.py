#!/usr/bin/env python3
"""
Warm-Start Module for SOTA Video Enhancement
"""

import torch
import logging

# Import all model handlers
from models.enhancement.vsr.vsrm_handler import VSRMHandler
from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
from models.enhancement.zeroshot.ditvr_handler import DiTVRHandler
from models.enhancement.vsr.fast_mamba_vsr_handler import FastMambaVSRHandler

logger = logging.getLogger(__name__)

def warm_start():
    """Preload models and JIT compile hot paths."""
    logger.info("🔥 Starting warm-start sequence...")

    # Dummy tensor for JIT compilation
    dummy_input = torch.rand(1, 3, 3, 64, 64).cpu()

    try:
        # VSRM
        vsrm = VSRMHandler(device='cpu')
        vsrm.model(dummy_input)
        logger.info("✅ VSRM handler warmed up")

        # SeedVR2
        seedvr2 = SeedVR2Handler(device='cpu')
        seedvr2.model(dummy_input)
        logger.info("✅ SeedVR2 handler warmed up")

        # DiTVR
        ditvr = DiTVRHandler(device='cpu')
        ditvr.model(dummy_input)
        logger.info("✅ DiTVR handler warmed up")

        # Fast-Mamba VSR
        fast_mamba = FastMambaVSRHandler(device='cpu')
        fast_mamba.model(dummy_input)
        logger.info("✅ Fast-Mamba VSR handler warmed up")

        logger.info("🎉 Warm-start sequence completed successfully!")

    except Exception as e:
        logger.error(f"❌ Warm-start sequence failed: {e}")

if __name__ == '__main__':
    warm_start()