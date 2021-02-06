import pprint
import numpy as np
from .utils import utils

try:
    import imageio as imageio
except ImportError:
    imageio = None

class OptimizerCallback:
    """Abstract class for defining callbacks for use with [Optimizer.minimize](vis.optimizer.md#optimizerminimize).
    """

    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        """This function will be called within [optimizer.minimize](vis.optimizer.md#minimize).

        Args:
            i: The optimizer iteration.
            named_losses: List of `(loss_name, loss_value)` tuples.
            overall_loss: Overall weighted loss.
            grads: The gradient of input image with respect to `wrt_value`.
            wrt_value: The current `wrt_value`.
        """
        raise NotImplementedError()
        
    def on_end(self):
        """Called at the end of optimization process. This function is typically used to cleanup / close any
        opened resources at the end of optimization.
        """
        pass


class Print(OptimizerCallback):
    """Callback to print values during optimization.
    """
    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        print(f"Iteration: {i+1}, named_losses: {pprint.pformat(named_losses)}, overall loss: {overall_loss}")
