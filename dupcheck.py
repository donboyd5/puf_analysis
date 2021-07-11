
import pandas as pd

DIR_FOR_BOYD_PUFCSV = r'/media/don/data/puf_files/puf_csv_related_files/Boyd/2021-07-02/'
PUFPATH = DIR_FOR_BOYD_PUFCSV + 'puf.csv'

puf = pd.read_csv(PUFPATH)

# check for duplicates using ALL columns other than RECID
cols = puf.columns.tolist()
cols.remove('RECID')
dups = puf[puf.duplicated(subset=cols, keep=False)] # keep = False keeps all duplicate records
dups.shape  # 2673 duplicates

# also drop FLPDYR
cols2 = cols.copy()
cols2.remove('FLPDYR')
dups2 = puf[puf.duplicated(subset=cols2, keep=False)]
dups2.shape  # 2828 duplicates

# which h_seq has the most duplicates?
dups2.h_seq.value_counts()