import sklearn.base
from abc import abstractmethod, ABC
from sklearn.utils import check_array, check_random_state
import pandas as pd
import os
import pickle
import numpy as np
import warnings  
import logging

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import keras
        from keras.backend import clear_session
    except (ModuleNotFoundError, ImportError) as e:
        pass
    try:
        import tensorflow.keras as keras
        from tensorflow.keras.backend import clear_session
    except (ModuleNotFoundError, ImportError) as e:
        raise ModuleNotFoundError('Could not find a working distribution of Keras and Tensorflow! Please install to use this module.')

_DEFAULT_PREDICTION_BATCH_SIZE = 50000


### Abstract base class for AD models

class AnomalyDetectionBase(sklearn.base.BaseEstimator, ABC):
    """base class which inherits from the sklearn base estimator. Provides broad functionality for 
    declaring new model classes, including defining save and load functions.
    
    A subclass should provide the following:

    1. an __init__ method which passes all relevant estimator parameters to the 
    _inputs_to_attributes function, most easily by way of calling locals()

    2. A function override for the "fit" function, taking as inputs the arrays
        x, y_sim, y_sr, w, and m, and returning a reference to the class instance (self)
    
    3. A function override for the "predict" function, taking as inputs an x-array
        and returning an array of predictions
    
    4. A function override for the "_model_names" function, returning a list of the 
        names of all keras models used for the estimator. This ensures proper saving

    Additional helper functions may be defined as necessary, though good practice is
    to prefix them with "_" to avoid namespace pollution.
    """

    # 
    # ADDED FUNCTIONALITY (saving and loading)
    # 

    def save(self, path, mkdirs=True,):
        """
        Save estimator information to a directory
        Parameters
        ----------
        path : str, required
            specifies the directory in which to save the model
        mkdirs : bool, default=True
            whether or not to create the directory given, if it doesn't exist
        Returns
        -------
        self : class instance
            useful for cascading object calls
        """        
        if not os.path.exists(path):
            if mkdirs:
                os.makedirs(path)
            else:
                raise FileNotFoundError('pathname "{}" not found. Set <mkdirs=True> to create directories.'.format(path))
        save_dict = self.get_params(exact_models=True)
        save_dict['classname'] = self.__class__.__name__

        models = dict()
        for k in list(save_dict.keys()):
            if k in self._model_names():
                models[k] = save_dict.pop(k)

        for k,model in models.items():
            mpath = '{}/{}'.format(path,k)
            model = _validate_model(model, k)
            save_dict[k] = model.to_json()
            model.save(mpath)

        with open('{}/params.pkl'.format(path), 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self

    def load(self, path):
        """
        Load saved information into estimator from a file
        Parameters
        ----------
        path : str, required
            specifies a directory in which the desired model was saved.
            should have at least the "params.pkl" pickle file in it.
        Returns
        -------
        self : class instance
            useful for cascading object calls
        """

        save_dict, classname = self._load_params(path)

        if classname != self.__class__.__name__:
            raise ValueError('tried to load savefile of class "{}" into object with class "{}"!'.format(classname, self.__class__.__name__))
        
        self._load_models(path, save_dict, classname)

        return self

    def get_params(self, deep=True, copy_models=False, exact_models=False):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        copy_models : bool, default=False
            If True, will return an identical (but copied) keras model
            as the ones specified by the class variables of the return of _model_names.
            Preserves model weights.
        exact_models : bool, default=False
            If True, will return the exact keras model as the one specified 
            by the class variables of the return of _model_names. Overrides
            parameter <copy_models>. If both copy_models and exact_models are false,
            then the model is cloned using keras.models.clone_model.
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
            
            if key in self._model_names():
                if isinstance(value, str):
                    out[key] = value
                elif value is None:
                    out[key] = None
                else:
                    # then it is a keras model
                    if exact_models:
                        out[key] = value
                    elif copy_models:
                        out[key] = keras.models.model_from_json(value.to_json())
                        out[key].set_weights(value.get_weights())
                    else:
                        out[key] = keras.models.clone_model(value)
            else:
                out[key] = value
        return out
        
    def copy(self, copy_models=True, exact_models=False):
        """
        copy this model
        Parameters
        ----------
        copy_models : bool, default=False
            If True, will return an identical (but copied) keras model
            as the ones specified by the class variables of the return of _model_names.
            Preserves model weights.
        exact_models : bool, default=False
            If True, will return the exact keras model as the one specified 
            by the class variables of the return of _model_names. Overrides
            parameter <copy_models>. If both copy_models and exact_models are false,
            then the model is cloned using keras.models.clone_model.
        Returns
        -------
        __class__ : copy of this class
        """
        return self.__class__(**self.get_params(copy_models=copy_models, exact_models=exact_models))

    #
    # ABSTRACT METHODS (need to be redefined by derived classes)
    #

    @abstractmethod
    def fit(self):
        return self
    
    @abstractmethod
    def predict(self):
        return None
        
    @abstractmethod
    def _model_names(self):
        return []

    #
    # HELPER FUNCTIONS (just for init, basically)
    #

    def _inputs_to_attributes(self, local_variables):
        """
        Set local variable dictionary as attributes; lazy __init__
        Parameters
        ----------
        local_variables : dict, required
            dictionary of key value pairs to set as class attributes to this
            instance of self 
        Returns
        -------
        """
        for k,v in local_variables.items():
            if k != 'self':
                setattr(self, k, v)


    @staticmethod
    def _load_params(path):
        """
        Load saved information into estimator from a file
        Parameters. Useful for loading generic models.
        ----------
        path : str, required
            specifies a directory in which the desired model was saved.
            should have at least the "params.pkl" pickle file in it.
        Returns
        -------
        save_dict : dict
            dictionary of parameters from found file
        classname : str
            classname of class of saved model
        """

        if not os.path.exists(path):
            raise FileNotFoundError('pathname "{}" not found.'.format(path))
        
        pkl_path = '{}/params.pkl'.format(path)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError('file "{}" not found. cannot load model.'.format(pkl_path))
        with open(pkl_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        classname = save_dict.pop('classname')
        return save_dict, classname

    def _load_models(self, path, save_dict, classname):
        """
        Load saved information into estimator from a file
        Parameters
        ----------
        path : str, required
            specifies a directory in which the desired model was saved.
            should have at least the "params.pkl" pickle file in it.
        save_dict : dict, required
            dictionary of parameters from found file
        classname : str, required
            classname of class of saved model
        Returns
        -------
        self : class instance
            useful for cascading object calls
        """
        for k in self._model_names():
            model_path = '{}/{}'.format(path, k)
            # if k in save_dict:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
            elif k in save_dict:
                model = keras.models.model_from_json(save_dict[k])
            else:
                model = None
            save_dict[k] = model
    
        self.set_params(**save_dict)

        return self

    def __copy__(self):
        return self.copy(exact_models=False)

### Helper functions for class definitions

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

    # if not isinstance(model, keras.Model):
    #     raise AttributeError('model is not a keras.Model instance!')
        
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
        print('decoding')
        try:
            model = keras.models.model_from_json(model)
        except JSONDecodeError:
            raise ValueError('parameter <{}> with value "{}" could not be decoded.'.format(name, model))
    return model


### functions for user usage

def autoload(path, **kwargs):
    """
    Dynamically load a model without a specified base class. Class used must be included in 
    this module, or else specified by name in as a keyword argument.
    Parameters
    ----------
    path : str, required
        specifies a directory in which the desired model was saved.
        should have at least the "params.pkl" pickle file in it.
    Returns
    -------
    AnomalyDetectionBase derived class
        loaded model, if its correct classname could be found
    """
    params, classname = AnomalyDetectionBase._load_params(path)
    if classname in globals():
        return globals()[classname]()._load_models(path, params, classname)
    else:
        raise AttributeError('classname "{}" from model at path {} not known!'.format(classname, path))

### Predefined class definitions for common model types

class SALAD(AnomalyDetectionBase):
    def __init__(
        self, sb_model=None, model=None, 
        optimizer='adam', metrics=[], loss='binary_crossentropy', 
        epochs=10, sb_epochs=10, batch_size=1000, sb_batch_size=1000,
        compile=True, callbacks=[], test_size=0., verbose=False,
        dctr_epsilon=1e-5,
    ):
        self._inputs_to_attributes(locals())

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

        self._history = sb_hist, sr_hist
        return self

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

    def _model_names(self):
        return ['model', 'sb_model']

class DataVsSim(AnomalyDetectionBase):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], 
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0., verbose=False,
    ):
        self._inputs_to_attributes(locals())

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

        self._history = self.model.fit(
            x, y_sim,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )
        return self

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()

    def _model_names(self):
        return ['model']

class CWoLa(AnomalyDetectionBase):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], 
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0., verbose=False,
    ):
        self._inputs_to_attributes(locals())

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

        self._history = self.model.fit(
            x, y_sr,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w,
            verbose=self.verbose
        )

        return self

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()

    def _model_names(self):
        return ['model']

class SACWoLa(AnomalyDetectionBase):
    def __init__(
        self, model=None, optimizer='adam', metrics=[], lambda_=1.0,
        loss='binary_crossentropy', epochs=10, batch_size=1000,
        compile=True, callbacks=[], test_size=0., verbose=False,
    ):
        self._inputs_to_attributes(locals())

    def fit(
        self, x, y_sim=None, y_sr=None, w=None, m=None
    ):

        if y_sr is None:
            raise ValueError('parameter <y_sr> must hold signal region/sideband tags!')
        if y_sim is None:
            raise ValueError('parameter <y_sim> must hold simulation/data tags!')

        self.model = _validate_model(self.model, 'model')
        x, y_sim, y_sr, w = _check_training_params(self.model, x, y_sim, y_sr, w)
        
        w_sacwola = np.ones_like(y_sim)
        w_sacwola[y_sim == 0] = self.lambda_

        if w is not None:
            w_sacwola *= w

        y_sacwola = np.abs(y_sr.astype(int) - (~y_sim.astype(bool)).astype(int))
        
        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self._history = self.model.fit(
            x, y_sacwola,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=self.test_size,
            batch_size=int(self.batch_size),
            sample_weight=w_sacwola,
            verbose=self.verbose
        )

        return self

    def predict(
        self, x
    ):
        return self.model.predict(x, batch_size=_DEFAULT_PREDICTION_BATCH_SIZE).squeeze()

    def _model_names(self):
        return ['model']
