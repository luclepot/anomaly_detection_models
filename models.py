import sklearn.base
from sklearn.utils import check_array, check_random_state

class atlas_anomaly_detection_base(sklearn.base.BaseEstimator):
    """base class which inherits from the sklearn base estimator. Provides broad functionality for 
    declaring new model classes, including defining save and load functions.
    
    A few things are REQUIRED in the subclass for this class to work. In particular 
    <path> should be a directory for the model; it may not yet exist.
    """

    def save(self, mkdirs=True):
        if self.path is None:
            raise ValueError('class variable <path> is None!')
        
        if not os.path.exists(self.path):
            if mkdirs:
                os.makedirs(self.path)
            else:
                raise FileNotFoundError('pathname "{}" not found. Set <mkdirs=True> to create directories.'.format(self.path))
        
        return 0
    
    def load(self):
        if self.path is None:
            raise ValueError('class variable <path> is None!')

        if not os.path.exists(self.path):
            raise FileNotFoundError('pathname "{}" not found.'.format(self.path))
        
        return 0
    
class SALAD(atlas_anomaly_detection_base):
    def __init__(
        self, random_state=None, path=None,
        sb_model=None, sr_model=None, optimizer='adam',
        metrics=[], loss='binary_crossentropy'

    ):
        self.random_state = random_state
        self.path = path
        self.optimizer=optimizer
        self.metrics=metrics
        self.loss=loss

        self.sb_model=sb_model
        self.sr_model=sr_model
        self.sb_arch=None
        self.sr_arch=None
        
    def fit_sb(self, x, y, epochs=10, compile=True):
        self.random_state_sb = check_random_state(self.random_state)
        
    def fit_sr(self, x, y, epochs=10):
        self.random_state_sr = check_random_state(self.random_state)

    def fit(self, x_sb, y_sb, x_sr, y_sr, epochs_sb=10, epochs_sr=10):
        return 0