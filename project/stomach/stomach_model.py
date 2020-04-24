from model.segmentation import SegModel

class Model(SegModel):
    """Model class for stomach project."""
    
    def __init__(self, cfg, data, mode):
        super(Model, self).__init__(cfg, data, mode)

