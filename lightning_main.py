import pytorch_lightning as L

# Pytorch lightning module wrapper around a model
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model