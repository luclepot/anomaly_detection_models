# anomaly_detection_models

Repository with some useful anomaly detection model definitions.

## install

for latest stable version

    pip install anomaly_detection_models [--user]

for most recent version, use
    
    git clone git@github.com:luclepot/anomaly_detection_models.git
    pip install . [--user] 

with the ``--user`` argument specifying local installation.

## usage

import models directly or subclass `anomaly_detection_base` to make a new model (instructions in-source)

## example

see [`demos/test.ipynb`](/demos/test.ipynb) for an example. general usage is like sklearn, as

```
from anomaly_detection_models import SACWoLa

sacwola = SACWoLa(epochs=10, lambda_=1.2)
sacwola.fit(x, y_sim, y_sb)

pred = sacwola.predict(x_test)
```
