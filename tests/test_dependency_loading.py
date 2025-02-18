import os
import sys
import unittest
from unittest.mock import patch, MagicMock

class TestDependencyLoading(unittest.TestCase):
    def setUp(self):
        # Clear any cached imports before each test
        if 'exla.models.clip' in sys.modules:
            del sys.modules['exla.models.clip']
        if 'exla.models.deepseek_r1' in sys.modules:
            del sys.modules['exla.models.deepseek_r1']
    
    @patch('exla.utils.dependency_manager.is_package_installed')
    @patch('exla.utils.dependency_manager.install_package')
    @patch('exla.utils.device_detect.detect_device')
    def test_clip_cpu_dependencies(self, mock_detect, mock_install, mock_is_installed):
        # Setup mocks
        mock_detect.return_value = {
            'type': 'cpu',
            'capabilities': {
                'gpu_available': False,
                'cuda_available': False
            }
        }
        # Simulate clip packages not being installed
        mock_is_installed.side_effect = lambda x: False
        
        # Import and initialize CLIP
        from exla.models.clip import clip
        model = clip()
        
        # Verify dependencies were checked
        mock_is_installed.assert_any_call('clip-client')
        mock_is_installed.assert_any_call('clip-server')
        
        # Verify dependencies were installed
        mock_install.assert_any_call('clip-client')
        mock_install.assert_any_call('clip-server')
    
    @patch('exla.utils.dependency_manager.is_package_installed')
    @patch('exla.utils.dependency_manager.install_package')
    @patch('exla.utils.device_detect.detect_device')
    def test_deepseek_gpu_dependencies(self, mock_detect, mock_install, mock_is_installed):
        # Setup mocks
        mock_detect.return_value = {
            'type': 'gpu',
            'capabilities': {
                'gpu_available': True,
                'cuda_available': True
            },
            'cuda_version': '118'
        }
        # Simulate no packages being installed
        mock_is_installed.return_value = False
        
        # Import and initialize DeepSeek
        from exla.models.deepseek_r1 import deepseek_r1
        model = deepseek_r1()
        
        # Verify base dependencies were checked
        mock_is_installed.assert_any_call('transformers')
        mock_is_installed.assert_any_call('accelerate')
        
        # Verify GPU-specific dependencies were checked
        mock_is_installed.assert_any_call('torch')
        mock_is_installed.assert_any_call('vllm')
        
        # Verify correct dependencies were installed
        mock_install.assert_any_call('transformers>=4.34.0')
        mock_install.assert_any_call('accelerate')
        mock_install.assert_any_call('torch>=2.0.0+cu118')
        mock_install.assert_any_call('vllm')
    
    @patch('exla.utils.dependency_manager.is_package_installed')
    @patch('exla.utils.dependency_manager.install_package')
    @patch('exla.utils.device_detect.detect_device')
    def test_no_reinstall_if_present(self, mock_detect, mock_install, mock_is_installed):
        # Setup mocks
        mock_detect.return_value = {
            'type': 'cpu',
            'capabilities': {
                'gpu_available': False,
                'cuda_available': False
            }
        }
        # Simulate all packages being already installed
        mock_is_installed.return_value = True
        
        # Import and initialize CLIP
        from exla.models.clip import clip
        model = clip()
        
        # Verify dependencies were checked
        mock_is_installed.assert_any_call('clip-client')
        mock_is_installed.assert_any_call('clip-server')
        
        # Verify no installations were attempted
        mock_install.assert_not_called()

    @patch('exla.utils.dependency_manager.install_package')
    def test_failed_installation(self, mock_install):
        # Simulate a failed installation
        mock_install.side_effect = Exception("Failed to install package")
        
        with self.assertRaises(RuntimeError) as context:
            from exla.models.clip import clip
            model = clip()
        
        self.assertIn("Failed to install", str(context.exception))

if __name__ == '__main__':
    unittest.main() 