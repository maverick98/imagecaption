from src.model.data.image_caption_dataset_sequential import ImageCaptionDataset
class Train:
    def __init__(self,imageCaptionDataset, **kwargs):
        super().__init__(**kwargs)
        self.imageCaptionDataset= imageCaptionDataset

    def train(self):
        self.imageCaptionDataset.train_model()
    

