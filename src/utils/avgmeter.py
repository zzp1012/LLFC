import pandas as pd
import torch
from collections import defaultdict
from typing import Optional

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """
    def __init__(self, delimiter='\n\t'):
        self.delimiter = delimiter
        self.reset()

    def reset(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, input_dict: dict, n: int=1):
        """Update the meter with a dictionary.

        Args:
            input_dict (dict): A dictionary of metrics.
            n (int): The number of samples in the input.
        """
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v, n)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter + self.delimiter.join(output_str)


class MetricTracker(object):
    """track metrics over time and compute average
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """reset metrics
        """
        self.metrics = defaultdict(list)
        self.meter = MetricMeter()

    def update(self, input_dict: dict, n: int=1):
        """update metrics

        Args:
            input_dict (dict): A dictionary of metrics.
            n (int): The number of samples in the input.
        """
        self.meter.update(input_dict, n)

    def track(self, input_dict: Optional[dict]=None):
        """track metrics
        """
        if input_dict is not None:
            for k, v in input_dict.items():
                assert k not in self.meter.meters, \
                    f'key {k} already exists in meter.keys() {self.meter.meters.keys()}'
                self.metrics[k].append(v)

        for k, v in self.meter.meters.items():
            self.metrics[k].append(v.avg)
        self.meter.reset()

    def __str__(self):
        return str(self.meter)
    
    def save_to_csv(self, filename: str):
        """save metrics to csv file
        """
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(filename, index=False)