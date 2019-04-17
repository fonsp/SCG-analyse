class Cluster:
    def __init__(self, location_range=None, time_range=None):
        """
        :param location_range: (Optional) Tuple containing left and right cluster bounds
        :param time_range: (Optional) Tuple containing start and end dates of cluster
        """
        self.location_range = location_range
        self.time_range = time_range

    def __str__(self):
        sentences = []
        if self.location_range is not None:
            sentences.append("{0:.0f}m to {1:.0f}m".format(*self.location_range))
        if self.time_range is not None:
            sentences.append("{0} until {1}".format(*self.time_range))
        return "; ".join(sentences)
