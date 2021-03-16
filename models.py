import sklearn.base
from abc import abstractmethod, ABC
from sklearn.utils import check_array, check_random_state

class anomaly_detection_base(sklearn.base.BaseEstimator, ABC):
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

    @abstractmethod
    def fit(self, x, y):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    def _inputs_to_attributes(self, local_variables):
        for k,v in local_variables.items():
            if k != 'self':
                setattr(self, k, v)
        

def _check_array_type(x):
    if isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    raise AttributeError('input array is of type "{}"; should be array'.format(type(x)))

def _check_training_params(model, x, y, w=None):
    """
    checks training dataset parameters x, y, and w against model <model>, including shapes and types
    """
    x,y = _check_array_type(x), _check_array_type(y)
    
    if not isinstance(model, keras.Model):
        raise AttributeError('model is not a keras.Model instance!')
    
    x_shape, y_shape = x.shape,y.shape
    
    input_match = model.input_shape[1] == np.array(x.shape)
    output_match = model.output_shape[1] == np.array(y.shape)
    
    if not input_match.any():
        raise AttributeError('x array shape {} does not match model input shape {}'.format(x_shape, model.input_shape))
    if not output_match.any():
        if len(output_match) == 1:
            output_match = np.array([True])
        else:
            raise AttributeError('y array shape {} does not match model output shape {}'.format(y_shape, model.output_shape))

    if len(input_match) > 2:
        raise AttributeError('input array must have less than 3 dimensions')
    if len(output_match) > 2:
        raise AttributeError('output array must have less than 3 dimensions')
    
    if np.where(input_match)[0][0] == 0:
        x = x.T
    if np.where(output_match)[0][0] == 0:
        y = y.T
        
    if w is not None:
        w = _check_array_type(w)
        if w.shape != y.shape:
            if w.T.shape != y.shape:
                raise AttributeError('weight array shape {} must match y array shape {}'.format(y.shape, w.shape))
            else:
                w = w.T
    return x, y, w

class SALAD(anomaly_detection_base):
    def __init__(
        self, path=None, sb_model=None, sr_model=None, 
        optimizer='adam', metrics=[], loss='binary_crossentropy', 
        sr_epochs=10, sb_epochs=10, compile=True, callbacks=[], 
        sb_arch=None, sr_arch=None, test_size=0.3,
    ):
        self._inputs_to_attributes(locals())

    def fit_sb(self, x, y, w=None):
        if self.sb_model is None:
            raise ValueError('parameter <sb_model> is None. Please set it to a valid keras model.')
        
        x, y, w = _check_training_params(self.sb_model, x, y, w)
        self.sb_model.fit(
            x, y,
        )
        return 0

    def fit_sr(self, x, y, w=None):
        if self.sr_model is None:
            raise ValueError('parameter <sb_model> is None. Please set it to a valid keras model.')
        
        x, y, w = _check_training_params(self.sr_model, x, y, w)
        self.sr_model.fit(
            x, y,
        )
        return 0 

    def fit(self, x_sb, y_sb, x_sr, y_sr):
        
        self.fit_sb(x_sb, y_sb)
        self.fit_sr(x_sr, y_sr)

        return 0

    def predict(self, x):
        raise NotImplementedError()