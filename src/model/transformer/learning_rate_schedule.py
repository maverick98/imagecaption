from tensorflow import math, cast, float32
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


#import gensim



class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LRScheduler, self).__init__()

        self.d_model = d_model
        self.d_model = cast(self.d_model, float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = cast(step, float32)
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return math.rsqrt(self.d_model) * math.minimum(arg1, arg2)
    