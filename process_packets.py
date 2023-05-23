import pandas as pd
import numpy as np

# df = pd.read_csv('impl_output.txt', delimiter='\t', header=None)
df = pd.read_csv('fragmentation_wireshark.txt', delimiter='\t', header=None)

cols = ['frame.time_delta', 'frame.len', 'radiotap.datarate',
        'radiotap.channel.type.cck', 'radiotap.dbm_antsignal',
        'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.protected',
        'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.ds',
        'wlan.fc.type', 'wlan.fc.subtype'] 
df.columns = cols

df['wlan.frag'].fillna(df['wlan.frag'].mode()[0], inplace=True)
df['wlan.seq'].fillna(df['wlan.seq'].mode()[0], inplace=True)
# df = df[df['radiotap.dbm_antsignal'].notna()]
df['radiotap.dbm_antsignal'].fillna(df['radiotap.dbm_antsignal'].mode()[0], inplace=True)
df['radiotap.channel.type.cck'].fillna(df['radiotap.channel.type.cck'].mode()[0], inplace=True)

# make type, subtype, and ds one-hot-encoded
wlan_fc_ds = pd.get_dummies(df['wlan.fc.ds'], prefix='wlan.fc.ds')
df = df.drop('wlan.fc.ds', axis=1)
wlan_fc_type = pd.get_dummies(df['wlan.fc.type'], prefix='wlan.fc.type')
df = df.drop('wlan.fc.type', axis=1)
wlan_fc_subtype = pd.get_dummies(df['wlan.fc.subtype'], prefix='wlan.fc.subtype')
df = df.drop('wlan.fc.subtype', axis=1)
"""
# no subtype 4 in just deauth packets, so insert all zeros
wlan_fc_subtype_4 = pd.Series([0 for i in range(len(df))], name='wlan.fc.subtype_4')
"""
df = pd.concat([df, wlan_fc_ds, wlan_fc_type, wlan_fc_subtype], axis=1)

# take last value of all dbm_antsignal
lm2 = lambda x : x[-3:] if (type(x) == str) else x
lm = lambda x: int(x) if x[0] != ',' else int(x[-2:])
df['radiotap.dbm_antsignal'] = df['radiotap.dbm_antsignal'].apply(lm2)
df['radiotap.dbm_antsignal'] = df['radiotap.dbm_antsignal'].apply(lm)

df = df.astype({'radiotap.dbm_antsignal': np.float64, 'wlan.duration': np.float64})

shortened_cols = ['frame.time_delta', 'frame.len', 'radiotap.datarate',
                    'radiotap.channel.type.cck', 'radiotap.dbm_antsignal',
                    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.protected',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.ds_0x01',
                    'wlan.fc.type_1', 'wlan.fc.subtype_0', 'wlan.fc.subtype_4',
                    'wlan.fc.subtype_5']
df = df[shortened_cols]

"""
fld_cols = ['frame.time_delta', 'frame.len', 'radiotap.dbm_antsignal',
            'wlan.fc.retry', 'wlan.duration', 'wlan.seq', 'wlan.fc.type_0',
            'wlan.fc.subtype_4']

df = df[fld_cols]
"""
df.info()

# Load model and put df through that
import lightgbm as lgb
import pickle

"""
with open("implementation_models/impinj_vs_fld_vs_normal.pkl", "rb") as f:
    model = pickle.load(f)
preds = model.predict(df.values)
"""

with open("implementation_models/injection_vs_rest.pkl", "rb") as f:
    model = pickle.load(f)
preds = model.predict(df.values)

# mapping = ['normal', 'imp/inj', 'flooding']
mapping = ['normal', 'injection']
class_pred = np.array([mapping[i] for i in preds])
unique, counts = np.unique(class_pred, return_counts=True)
print(class_pred)
print(np.asarray((unique, counts)).T)

"""
y_pred_prob = model.predict_proba(df.values)
y_pred = []
for i in range(y_pred_prob.shape[0]):
    if y_pred_prob[i, 1] > 0.01:
        y_pred.append(1)
    else:
        y_pred.append(0)
class_pred = np.array([mapping[i] for i in y_pred])
unique, counts = np.unique(class_pred, return_counts=True)
print(np.asarray((unique, counts)).T)
"""
"""
print(class_pred[212720])
print(class_pred[212721])
"""
