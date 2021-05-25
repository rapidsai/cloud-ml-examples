#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml import Ridge
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient
from cuml.model_selection import train_test_split
from azureml.core.run import Run
import joblib
import os
import numpy as np
import time
import cudf
import cuml
import interpret_community
from cuml.benchmark.datagen import load_higgs

from datetime import datetime
from dateutil import parser

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = load_higgs()
N_ROWS = 1000000
run = Run.get_context()
client = ExplanationClient.from_run(run)
run.log('N_ROWS', N_ROWS)
X_train, X_test, y_train, y_test = train_test_split(X[:N_ROWS],
                                                    y[:N_ROWS],
                                                    random_state=1)
# write x_test out as a pickle file for later visualization
x_test_pkl = 'x_test.pkl'
with open(x_test_pkl, 'wb') as file:
    joblib.dump(value=X_test, filename=os.path.join(OUTPUT_DIR, x_test_pkl))
run.upload_file('x_test_higgs.pkl', os.path.join(OUTPUT_DIR, x_test_pkl))


gamma = 0.001
C = 100.
# Use Ridge algorithm to create a regression model
import cuml
reg = cuml.svm.SVC(C=C, gamma=gamma, probability=True)
model = reg.fit(X_train, y_train)

# preds = reg.predict(X_test)
run.log('C', C)
run.log('gamma', gamma)


model_file_name = 'svc.pkl'
# save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=reg, filename=os.path.join(OUTPUT_DIR,
                                                 model_file_name))

# register the model
run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='model_explain_model_on_amlcomp',
                                    model_path='original_model.pkl')

# Explain predictions on your local machine
tabular_explainer = TabularExplainer(model, X_train.to_pandas(), features=X_train.columns, use_gpu=True)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
global_explanation = tabular_explainer.explain_global(X_test.to_pandas()[:50])

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
comment = 'Global explanation on regression model trained on boston dataset'
client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)