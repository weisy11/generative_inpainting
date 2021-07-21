""" run discriminator """
import time
import tensorflow as tf

from . import PeriodicCallback, CallbackLoc
from ..train.multigpu_trainer import MultiGPUTrainer


class SecondaryMultiGPUTrainer(PeriodicCallback, MultiGPUTrainer):

    """SecondaryMultiGPUTrainer.

    """

    def __init__(self, pstep, **context):
        PeriodicCallback.__init__(self, CallbackLoc.step_start, pstep)
        context['log_progress'] = context.pop('log_progress', False)
        MultiGPUTrainer.__init__(self, primary=False, **context)

    def run(self, sess, step):
        self.context['sess'] = sess
        self.train()
