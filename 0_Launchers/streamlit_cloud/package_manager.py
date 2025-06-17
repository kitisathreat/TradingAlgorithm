import pkg_resources
import subprocess
import sys
import logging
import os
from typing import Dict, Optional, Tuple, List
from packaging import version
from packaging.specifiers import SpecifierSet
import platform

def is_streamlit_cloud() -> bool:
    """
    Detect if running in Streamlit Cloud using multiple checks.
    Returns True if any of the checks indicate Streamlit Cloud.
    """
    # Check 1: Platform processor (empty string in Streamlit Cloud)
    if platform.processor() == '':
        return True
        
    # Check 2: Environment variable
    if os.environ.get('STREAMLIT_SERVER_RUNNING_ON_CLOUD', '').lower() == 'true':
        return True
        
    # Check 3: Mount path (exists in Streamlit Cloud)
    if os.path.exists('/mount/src'):
        return True
        
    # Check 4: Home directory (exists in Streamlit Cloud)
    if os.path.exists('/home/adminuser'):
        return True
        
    return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PackageManager:
    def __init__(self, cache_dir: str = "./.pip-cache"):
        """
        Initialize the package manager with a cache directory.
        
        Args:
            cache_dir: Directory to store pip cache. Defaults to ./.pip-cache
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self._installed_packages: Dict[str, pkg_resources.Distribution] = {}
        self._load_installed_packages()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using pip cache directory: {self.cache_dir}")

    def _load_installed_packages(self) -> None:
        """Load all currently installed packages into memory."""
        self._installed_packages = {
            pkg.key: pkg for pkg in pkg_resources.working_set
        }
        logger.debug(f"Loaded {len(self._installed_packages)} installed packages")

    def _parse_requirement(self, req_str: str) -> Tuple[str, Optional[str]]:
        """
        Parse a requirement string into package name and version specifier.
        
        Args:
            req_str: Requirement string (e.g., "numpy>=1.24.0")
            
        Returns:
            Tuple[str, Optional[str]]: (package_name, version_specifier)
        """
        try:
            # Handle comments and empty lines
            req_str = req_str.split('#')[0].strip()
            if not req_str:
                return "", None
                
            req = pkg_resources.Requirement.parse(req_str)
            return req.name, str(req.specifier) if req.specifier else None
        except Exception as e:
            logger.error(f"Error parsing requirement {req_str}: {e}")
            return req_str, None

    def _is_version_satisfied(self, package_name: str, version_spec: str) -> bool:
        """
        Check if the installed version satisfies the version specification.
        
        Args:
            package_name: Name of the package
            version_spec: Version specification (e.g., ">=1.24.0")
            
        Returns:
            bool: True if version is satisfied
        """
        if package_name not in self._installed_packages:
            return False

        installed_version = self._installed_packages[package_name].version
        spec = SpecifierSet(version_spec)
        return version.parse(installed_version) in spec

    def _get_cached_version(self, package_name: str) -> Optional[str]:
        """
        Get the version of a package from the cache if available.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            Optional[str]: Version string if found in cache, None otherwise
        """
        try:
            # Use pip to check cache
            result = subprocess.run(
                [sys.executable, "-m", "pip", "cache", "info", package_name],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                # Parse the output to get version
                for line in result.stdout.split('\n'):
                    if 'version' in line.lower():
                        return line.split(':')[-1].strip()
        except Exception as e:
            logger.warning(f"Error checking cache for {package_name}: {e}")
        return None

    def should_install_package(self, requirement: str) -> Tuple[bool, str]:
        """
        Determine if a package should be installed based on current version and cache.
        
        Args:
            requirement: Package requirement string (e.g., "numpy>=1.24.0")
            
        Returns:
            Tuple[bool, str]: (should_install, reason)
        """
        package_name, version_spec = self._parse_requirement(requirement)
        
        if not package_name:  # Skip empty lines or comments
            return False, "Empty requirement or comment"
        
        # Check if package is already installed with satisfactory version
        if package_name in self._installed_packages:
            if not version_spec or self._is_version_satisfied(package_name, version_spec):
                installed_version = self._installed_packages[package_name].version
                return False, f"Package {package_name} {installed_version} already installed and satisfies {version_spec or 'any version'}"
        
        # Check cache for available version
        cached_version = self._get_cached_version(package_name)
        if cached_version:
            if version_spec:
                spec = SpecifierSet(version_spec)
                if version.parse(cached_version) in spec:
                    return True, f"Found compatible version {cached_version} in cache"
            else:
                return True, f"Found version {cached_version} in cache"
        
        return True, f"Package {package_name} needs to be installed or updated"

    def install_package(self, requirement: str, use_cache: bool = True) -> bool:
        """
        Install a package if needed, using cache when available.
        
        Args:
            requirement: Package requirement string
            use_cache: Whether to use cached packages when available
            
        Returns:
            bool: True if installation was successful or not needed
        """
        should_install, reason = self.should_install_package(requirement)
        
        if not should_install:
            logger.info(reason)
            return True

        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if use_cache:
                cmd.extend(["--cache-dir", self.cache_dir])
            cmd.append(requirement)
            
            logger.info(f"Installing {requirement}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully installed {requirement}")
            self._load_installed_packages()  # Refresh installed packages
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {requirement}: {e.stderr}")
            return False

    def install_requirements(self, requirements_file: str, use_cache: bool = True) -> List[str]:
        """
        Install packages from a requirements file.
        
        Args:
            requirements_file: Path to requirements.txt file
            use_cache: Whether to use cached packages when available
            
        Returns:
            List[str]: List of failed installations
        """
        failed_installations = []
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Installing {len(requirements)} packages from {requirements_file}")
            
            for requirement in requirements:
                if not self.install_package(requirement, use_cache):
                    failed_installations.append(requirement)
            
            if failed_installations:
                logger.error(f"Failed to install {len(failed_installations)} packages: {failed_installations}")
            else:
                logger.info("All packages installed successfully")
                
        except FileNotFoundError:
            logger.error(f"Requirements file not found: {requirements_file}")
            failed_installations.append(f"File not found: {requirements_file}")
        except Exception as e:
            logger.error(f"Error reading requirements file: {e}")
            failed_installations.append(f"Error: {e}")
        
        return failed_installations

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package Manager for Trading Algorithm")
    parser.add_argument("requirements_file", help="Path to requirements.txt file")
    parser.add_argument("--no-cache", action="store_true", help="Don't use pip cache")
    
    args = parser.parse_args()
    
    manager = PackageManager()
    failed = manager.install_requirements(args.requirements_file, not args.no_cache)
    
    if failed:
        sys.exit(1)
    else:
        print("All packages installed successfully")

if __name__ == "__main__":
    main() 