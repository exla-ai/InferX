"""
This script demonstrates the lazy loading of dependencies for different models.
Run it with different combinations to see how dependencies are loaded.
"""

def test_clip():
    print("\n=== Testing CLIP Model ===")
    from exla.models import clip
    model = clip()
    print("CLIP model initialized successfully")

def test_deepseek():
    print("\n=== Testing DeepSeek Model ===")
    from exla.models import deepseek_r1
    model = deepseek_r1()
    print("DeepSeek model initialized successfully")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_lazy_loading.py [clip|deepseek|all]")
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    if test_type == 'clip':
        test_clip()
    elif test_type == 'deepseek':
        test_deepseek()
    elif test_type == 'all':
        test_clip()
        test_deepseek()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available options: clip, deepseek, all") 