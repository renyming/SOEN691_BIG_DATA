"""
References:
https://www.kaggle.com/kerneler/starter-network-intrusion-detection-68f11311-8#
http://alanpryorjr.com/visualizations/seaborn/heatmap/heatmap/

This file is proposed to generate correlations between features.
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def preprocessing(df, dfname):
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    df.dataframeName = dfname
    nRow, nCol = df.shape
    print('There are %d rows and %d columns in \"%s\"' % (nRow, nCol, dfname))
    return df


def plot_correlation(df):
    corr = df.corr()
    plt.figure(figsize=(25, 25))
    sn.heatmap(corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 10},
        cbar=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    plt.title('Correlation Matrix for %s' % df.dataframeName, fontsize=25)
    plt.gca().xaxis.tick_bottom()
    plt.show()


headers = ["duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "target"]

# plot the correlation in Train.csv
df_train_orig = pd.read_csv('./source_dir/Train.csv', header=None, names=headers)
df_train_orig = preprocessing(df_train_orig, 'Train.csv')
plot_correlation(df_train_orig)

# plot the correlation in Train_clean.csv
df_train_clean = pd.read_csv('./source_dir/Train_clean.csv', header=None, names=headers)
df_train_clean = preprocessing(df_train_clean, 'Train_clean.csv')
plot_correlation(df_train_clean)

