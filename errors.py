
import numpy as np
import pickle

qtiles = [0.0, .1, .25, .5, .75, .9, 1.0]




save_name = '/home/donboyd/Documents/beta_xmat.pkl'
open_file = open(save_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

b, xm = pkl


b.shape
xm.shape

# beta_x = np.exp(np.dot(beta, xmat.T))
bx = np.dot(b, xm.T)
bx.shape

np.quantile(b, qtiles)
np.quantile(xm, qtiles)
np.quantile(bx, qtiles)
ebx = np.exp(bx)
np.quantile(ebx, qtiles)

np.exp(-300000.)
np.exp(100.)

np.argmax(bx)
bx.flatten()[np.argmax(bx)]

# save_list = [bvec, wh, xmat, geotargets, dw, diffs]
save_name = '/home/donboyd/Documents/bvwhxmgtdwdf.pkl'

open_file = open(save_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

bvec, wh, xmat, geotargets, dw, diffs = pkl
np.quantile(bvec, qtiles)
# array([-20129.5591439 ,  -3921.41872121,  -1303.89841512,    -47.48800578,
#           933.14636517,   2860.35694082,  23683.59500333])
bvec[(bvec > -0.1) & (bvec < 0.1)]

np.quantile(diffs, qtiles)




# np.quantile(reslsq.method_result.beta_opt, qtiles)
# array([-6441.40456227, -5301.09844519, -1045.63633219,  -441.84611902,
#         -167.47558302,    23.80319664,   274.96759786,   875.14966463,
#         1157.70746797,  1716.71928114,  4188.58560741])