from datetime import datetime


class MC:
    """
    Micro-Cluster in MCNN
    """

    def __init__(self, theta):
        self.epsilon = 0            # error counter
        self.theta = theta          # upper error threshold
        self.centroid = None        # centroid
        self.cf1_x = None           # sum of feature values
        self.cf1_t = None           # sum of time stamps
        self.cf2_x = None           # sum of squares of attributes
        self.n = 0                  # no of instances
        self.cl = 'unknown'         # class label
        self.alpha = datetime.now() # initial time stamp
        self.omega = 0              # threshold for performance

        # attributes for the split purpose
        self.variance_x = []        # variance vector for each attribute
        self.status = 'unknown'     # indicate if the mc is activate
        self.filename = 'unknown.csv'

