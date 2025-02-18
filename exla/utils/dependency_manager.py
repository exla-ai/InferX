import importlib
import subprocess
import sys
from typing import List, Optional

def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name.split('>=')[0].split('==')[0])
        return True
    except ImportError:
        return False

def install_package(package_name: str):
    """Install a package using pip."""
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "--quiet",
        package_name
    ])

def ensure_dependencies(model_name: str, packages: List[str], hardware_specific: Optional[List[str]] = None):
    """
    Ensures all required dependencies for a model are installed.
    
    Args:
        model_name: Name of the model (for error messages)
        packages: List of required packages
        hardware_specific: Optional list of hardware-specific packages
    """
    missing_packages = [
        pkg for pkg in packages 
        if not is_package_installed(pkg.split('>=')[0].split('==')[0])
    ]
    
    if hardware_specific:
        missing_packages.extend([
            pkg for pkg in hardware_specific 
            if not is_package_installed(pkg.split('>=')[0].split('==')[0])
        ])
    
    if missing_packages:
        print(f"\n📦 Installing required dependencies for {model_name}...")
        for package in missing_packages:
            print(f"   • Installing {package}")
            try:
                install_package(package)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to install {package} for {model_name}. "
                    f"You can install it manually with: pip install {package}"
                ) from e
        print(f"✨ Successfully installed all dependencies for {model_name}\n") 