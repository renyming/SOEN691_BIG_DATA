from datetime import datetime

class MC:
    """
    Micro-Cluster in MCNN
    """

    def __init__(self, theta):
        self.epsilon = 0            # error counter
        self.theta = theta          # upper threshold
        self.centroid = None        # centroid
        self.cf_all = []            # cluster feature (all, used for writing) #may not used this
        self.cf = []                # cluster feature (numeric only)          #and this
        self.cf1_x = None           # sum of feature values
        self.cf1_t = None           # sum of time stamps
        self.cf2_x = None           # sum of squares of attributes
        self.n = 0                  # no of instances
        self.cl = 'unknown'         # class label
        self.alpha = datetime.now() # initial time stamp
        self.omega = 0              # threshold for performance

    def print(self):
        print("epsilon   = " + str(self.epsilon) + "\n" + \
              "theta     = " + str(self.theta) + "\n" + \
              "# instances   = " + str(self.n) + "\n" + \
              "class label   = " + self.cl + "\n" + \
              "timestamp = " + str(self.alpha) + "\n" + \
              "omega     = " + str(self.omega))
        print("centroid  = ", self.centroid)
        print("features  = ", self.cf)
        print("features sum  = ", self.cf1_x)
        print("timestamp sum = ", self.cf1_t)
        print("features soq  = ", self.cf2_x)