# estimation package contains different proficiency estimation methods
from catsim.estimation import *
from catsim.initialization import *
from catsim.selection import *
from catsim.stopping import *


class Irt:
    def __init__(self, item_bank):
        self.estimator = NumericalSearchEstimator()
        self.selector = MaxInfoSelector()
        self.stopper = MaxItemStopper(10)
        self.item_bank = item_bank

    def estimate_theta(self, administered_items=None, responses=None, est_theta=None):
        if administered_items is None:
            initializer = FixedPointInitializer(2)
            new_theta = initializer.initialize()
        else:
            new_theta = self.estimator.estimate(items=self.item_bank, administered_items=administered_items,
                                                response_vector=responses, est_theta=est_theta)
        return new_theta

    def next_item(self, est_theta, administered_items=None):
        if administered_items is None:
            return 0
        else:
            item_index = self.selector.select(items=self.item_bank, administered_items=administered_items,
                                              est_theta=est_theta)
            print('Next item to be administered:', item_index)
            return item_index
