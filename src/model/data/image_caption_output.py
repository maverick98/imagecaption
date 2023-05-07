class ImageCaptionOutput:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.caption=None
        self.caption_probabilities=None
        self.attention_weights=None
        self.joint_probability=0.0
        self.initialized=False
        self.output=None