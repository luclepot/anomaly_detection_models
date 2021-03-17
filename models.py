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

    def get_params(self, deep=True, copy_models=False):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            
            if key in self._MODEL_NAMES:
                if isinstance(value, str):
                    out[key] = value
                else:
                    # then it is a keras model
                    if copy_models:
                        out[key] = keras.models.clone_model(value)
                    else:
                        out[key] = value.to_json()
            else:
                out[key] = value
        return out
        
def _check_array_type(x):
    if isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    raise AttributeError('input array is of type "{}"; should be array'.format(type(x)))

def _check_training_params(model, x, *y_args):
    """
    checks training dataset parameters x, y, and w against model <model>, including shapes and types
    """
    x = _check_array_type(x)

    y_args = list(y_args)
    arg_shapes = []
    for i in range(len(y_args)):
        arg = y_args[i]
        if arg is not None:
            arg = _check_array_type(arg)
            if len(np.squeeze(arg).shape) > 1:
                raise AttributeError('one of the input y-style arrays is non-vector valued!')
            arg_shapes.append(arg.size)
        y_args[i] = arg

    if len(np.unique(np.array(arg_shapes))) > 1:
        raise AttributeError('input y value array shapes do not match')

    if not isinstance(model, keras.Model):
        raise AttributeError('model is not a keras.Model instance!')
        
    input_match = model.input_shape[1] == np.array(x.shape)
    
    if not input_match.any():
        raise AttributeError('x array shape {} does not match model input shape {}'.format(x.shape, model.input_shape))
    if len(input_match) > 2:
        raise AttributeError('input array must have less than 3 dimensions')
    
    if np.where(input_match)[0][0] == 0:
        x = x.T

    return tuple([x] + y_args)

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
        self, sb_model=None, model=None, 
        optimizer='adam', metrics=[], loss='binary_crossentropy', 
        epochs=10, sb_epochs=10, batch_size=1000, sb_batch_size=1000,
        compile=True, callbacks=[], test_size=0.3, verbose=False,
        dctr_epsilon=1e-5,
    ):
        self._inputs_to_attributes(locals())
        self._MODEL_NAMES = ['model', 'sb_model']

    def fit(
        self, x, y_sim=None, y_sr=None, w=None, m=None
    ):
        if y_sim is None:
            raise ValueError('parameter <y_sim> must hold simulation/data tags!')
        if y_sr is None:
            raise ValueError('parameter <y_sr> must hold signal region/sideband tags!')
        if m is None:
            raise ValueError('parameter <m> must be a localizing feature for SALAD!')
        
        sb_tag, sr_tag = ~y_sr.astype(bool), y_sr.astype(bool)
        sb_hist = self._fit_sb(x[sb_tag], y_sim[sb_tag], w=(w[sb_tag] if w is not None else w), m=m[sb_tag])
        sr_hist = self._fit_sr(x[sr_tag], y_sim[sr_tag], w=(w[sr_tag] if w is not None else w), m=m[sr_tag])

        return sb_hist, sr_hist

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()

    def predict_weight(
        self, x
    ):
        yhat = self.sb_model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE)
        return np.squeeze(yhat/(1 + self.dctr_epsilon - yhat))

    def _fit_sb(
        self, x, y_sim, w=None, m=None
    ):

        self.sb_model = _validate_model(self.sb_model, 'sb_model')
        if len(m.shape) < 2:
            m = m[:,np.newaxis]
        x = np.concatenate([m, x], axis=1)
        x, y_sim, w = _check_training_params(self.sb_model, x, y_sim, w)
        


        if self.compile:
            self.sb_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.sb_model.fit(
            x, y_sim,
            epochs=self.sb_epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.sb_batch_size),
            sample_weight=w,
            verbose=self.verbose
        )
        return 0

    def _fit_sr(
        self, x, y_sim, w=None, m=None
    ):

        self.model = _validate_model(self.model, 'model')
        self.sb_model = _validate_model(self.sb_model, 'sb_model')
        
        if len(m.shape) < 2:
            m = m[:,np.newaxis]
        x_dctr = np.concatenate([m, x], axis=1)

        x_dctr, = _check_training_params(self.sb_model, x_dctr)
        x, y_sim, w = _check_training_params(self.model, x, y_sim, w)
        
        w_dctr = self.predict_weight(x_dctr)
        w_dctr[y_sim == 1] = 1

        if w is not None:
            if w.shape != w_dctr.shape:
                raise AttributeError('given weight {} and DCTR weight {} do not match!'.format(w.shape, w_dctr.shape))
            w *= w_dctr
        else:
            w = w_dctr

        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.model.fit(
            x, y_sim,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )


class data_vs_sim(anomaly_detection_base):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], 
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0.3, verbose=False,
    ):
        self._inputs_to_attributes(locals())
        self._MODEL_NAMES = ['model']

    def fit(
        self, x, y_sim=None, y_sr=None, w=None, m=None
    ):
        if y_sim is None:
            raise ValueError('parameter <y_sim> must hold simulation/data tags!')
        
        self.model = _validate_model(self.model, 'model')
        x, y_sim, y_sr, w = _check_training_params(self.model, x, y_sim, y_sr, w)
        
        if y_sr is not None:
            x = x[y_sr == 1]
            if w is not None:
                w = w[y_sr == 1]
            y_sim = y_sim[y_sr == 1]

        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.model.fit(
            x, y_sim,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()

class cwola(anomaly_detection_base):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], 
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0.3, verbose=False,
    ):
        self._inputs_to_attributes(locals())
        self._MODEL_NAMES = ['model']

    def fit(
        self, x, y_sim=None, y_sr=None, w=None, m=None
    ):

        if y_sr is None:
            raise ValueError('parameter <y_sr> must hold signal region/sideband tags!')

        self.model = _validate_model(self.model, 'model')
        x, y_sim, y_sr, w = _check_training_params(self.model, x, y_sim, y_sr, w)
        
        if y_sim is not None:
            x = x[y_sim == 1]
            y_sr = y_sr[y_sim == 1]
            if w is not None:
                w = w[y_sim == 1]
        
        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return self.model.fit(
            x, y_sr,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()
