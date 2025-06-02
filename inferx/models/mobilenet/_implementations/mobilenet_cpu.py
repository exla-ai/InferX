from ._base import MobileNetBase

class MobileNetCPU(MobileNetBase):
    def __init__(self):
        print("Initializing CPU Model")