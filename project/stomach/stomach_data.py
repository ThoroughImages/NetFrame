import tensorflow as tf
from data.data_dealer import DataDealer


class Data(DataDealer):
    """Data input for stomach project."""

    def __init__(self, cfg, mode):
        super(Data, self).__init__(cfg, mode)

