import sklearn.base
from abc import abstractmethod, ABC
from sklearn.utils import check_array, check_random_state
import pandas as pd
import numpy as np

try:
    import keras
except ModuleNotFoundError:
    pass
try:
    import tensorflow.keras as keras
except ModuleNotFoundError:
    raise ModuleNotFoundError('Could not find a working distribution of Keras!')

_DEFAULT_PREDICTION_BATCH_SIZE = 50000

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

def _validate_model(model, name):
    if model is None:
        raise ValueError('parameter <{}> is None. Please set it to a valid keras model/keras json architecture.'.format(name))
    elif isinstance(model, str):
        try:
            model = keras.models.model_from_json(model)
        except JSONDecodeError:
            raise ValueError('parameter <{}> with value "{}" could not be decoded.'.format(name, model))
    return model

class SALAD(anomaly_detection_base):
    def __init__(
        self, sb_model=None, sr_model=None, 
        optimizer='adam', metrics=[], loss='binary_crossentropy', 
        sr_epochs=10, sb_epochs=10, sr_batch_size=1000, sb_batch_size=1000,
        compile=True, callbacks=[], test_size=0.3, verbose=False,
        dctr_epsilon=1e-5, m_cols=None
    ):
        self._inputs_to_attributes(locals())

    def fit_sb(self, x, y, w=None):

        self.sb_model = _validate_model(self.sb_model, 'sb_model')
        x, y, w = _check_training_params(self.sb_model, x, y, w)
        
        if self.compile:
            self.sb_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.sb_model.fit(
            x, y,
            epochs=self.sb_epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.sb_batch_size),
            sample_weight=w,
            verbose=self.verbose
        )
        return 0

    def fit_sr(self, x, y, w=None):

        self.sr_model = _validate_model(self.sr_model, 'sr_model')
        self.sb_model = _validate_model(self.sb_model, 'sb_model')
        
        if len(self.m_cols) == 0:
            raise AttributeError('Parameter <m_cols> should have at least one localizing feature column!')

        mask = np.ones(x.shape[1], dtype=bool)
        mask[self.m_cols] = False
        
        x_dctr,_,_ = _check_training_params(self.sb_model, x, y, w)
        x, y, w = _check_training_params(self.sr_model, x[:,mask], y, w)
        w_dctr = self.predict_weight(x_dctr)
        w_dctr[y == 1] = 1

        if w is not None:
            if w.shape != w_dctr.shape:
                raise AttributeError('given weight {} and DCTR weight {} do not match!'.format(w.shape, w_dctr.shape))
            w *= w_dctr
        else:
            w = w_dctr

        if self.compile:
            self.sr_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.sr_model.fit(
            x, y,
            epochs=self.sr_epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.sr_batch_size),
            sample_weight=w,
            verbose=self.verbose
        )

    def fit(self, x_sb, y_sb, x_sr, y_sr, w_sb=None, w_sr=None):
        
        sb_hist = self.fit_sb(x_sb, y_sb, w=w_sb)
        sr_hist = self.fit_sr(x_sr, y_sr, w=w_sr)

        return sb_hist,sr_hist

    def predict(self, x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE):
        mask = np.ones(x.shape[1], dtype=bool)
        mask[self.m_cols] = False
        return self.sr_model.predict(x[:,mask], batch_size=batch_size).squeeze()

    def predict_weight(self, x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE):
        yhat = self.sb_model.predict(x, batch_size=batch_size)
        return np.squeeze(yhat/(1 + self.dctr_epsilon - yhat))

class data_vs_sim(anomaly_detection_base):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], 
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0.3, verbose=False,
    ):
        self._inputs_to_attributes(locals())

    def fit(self, x, y, w=None):

        self.model = _validate_model(self.model, 'model')
        x, y, w = _check_training_params(self.model, x, y, w)
        
        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.model.fit(
            x, y,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )

    def predict(self, x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE):
        return self.model.predict(x, batch_size=batch_size).squeeze()
