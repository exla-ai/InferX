class DataLoader:
    def __init__(self, path):
        self.path = path
        
    def __iter__(self):
        # Implement data loading logic
        pass

def data_loader(path):
    return DataLoader(path) 