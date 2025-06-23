from src.nights import Nights


class BGAnalysis:
    """
    Class for analysing blood glucose data from nights.
    """
    def __init__(self, nights_objects: [Nights]):

        self.bg_data = bg_data