"""
Disk-based caching system for mesh data.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Any


class MeshCache:
    """
    Cache motor meshes to disk for faster repeated renders.
    
    Uses MD5 hash of geometry parameters as cache key.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize mesh cache.
        
        Args:
            cache_dir: Directory for cache files. Defaults to .cache/meshes
        """
        if cache_dir is None:
            cache_dir = Path('.cache/meshes')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, geometry) -> str:
        """
        Generate unique hash for motor geometry.
        
        Args:
            geometry: MotorGeometry instance
            
        Returns:
            MD5 hash string
        """
        # Serialize geometry parameters
        params = geometry.to_dict()
        params_str = str(sorted(params.items()))
        
        # Generate hash
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def load(self, cache_key: str) -> Optional[Any]:
        """
        Load mesh from cache.
        
        Args:
            cache_key: Hash key for cached mesh
            
        Returns:
            Cached mesh or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return None
        
        return None
    
    def save(self, cache_key: str, mesh: Any):
        """
        Save mesh to cache.
        
        Args:
            cache_key: Hash key for mesh
            mesh: Mesh object to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(mesh, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def clear(self):
        """Remove all cached meshes."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"Cleared {self.cache_dir}")
