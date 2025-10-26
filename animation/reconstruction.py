"""
PM Reconstruction - Convert PCA components back to movement data
"""

import numpy as np
from typing import List, Dict, Optional
from ppma_core import PrincipalMovementAnalysis, TrialKey


class PMReconstructor:
    """Reconstruct movement from PCA components"""
    
    def __init__(self, pma: PrincipalMovementAnalysis, preprocessor=None):
        self.pma = pma
        self.preprocessor = preprocessor
    
    def reconstruct_pm(
        self,
        pm_index: int,
        amplification: float = 2.0,
        n_frames: int = 60,
        trial_key: Optional[TrialKey] = None
    ) -> np.ndarray:
        """
        Reconstruct movement for a single PM
        Animation goes from -amp*SD through 0 to +amp*SD and back
        
        Args:
            pm_index: Which PM to reconstruct (0-indexed)
            amplification: Amplification factor (SD multiplier)
            n_frames: Number of frames in animation
            trial_key: Trial key for getting mean posture
            
        Returns:
            Array of shape (n_frames, n_coords) showing reconstructed motion
        """
        pc = self.pma.pcs_[:, pm_index]
        
        # Get standard deviation of this PM's activation
        if trial_key and trial_key in self.pma.pp_:
            pp_values = self.pma.pp_[trial_key][:, pm_index]
            sd = np.std(pp_values)
        else:
            first_key = list(self.pma.pp_.keys())[0]
            pp_values = self.pma.pp_[first_key][:, pm_index]
            sd = np.std(pp_values)
        
        # Create oscillation: Start at -amp*SD, go to +amp*SD, back to -amp*SD
        # This ensures animation starts at an extreme, not at zero!
        t = np.linspace(0, 2*np.pi, n_frames)
        pp_anim = amplification * sd * np.sin(t - np.pi/2)  # Start at minimum
        
        # Reconstruct: X = PP * PC^T
        reconstructed = np.outer(pp_anim, pc)
        
        # Add mean posture
        if self.preprocessor and trial_key and trial_key in self.preprocessor.mean_matrices:
            mean_posture = self.preprocessor.mean_matrices[trial_key]
            reconstructed += mean_posture
        
        return reconstructed
    
    def reconstruct_multiple_pms(
        self,
        pm_indices: List[int],
        amplifications: List[float],
        n_frames: int = 60,
        trial_key: Optional[TrialKey] = None
    ) -> Dict[int, np.ndarray]:
        """Reconstruct multiple PMs"""
        reconstructed = {}
        for pm_idx, amp in zip(pm_indices, amplifications):
            reconstructed[pm_idx] = self.reconstruct_pm(
                pm_idx, amp, n_frames, trial_key
            )
        return reconstructed