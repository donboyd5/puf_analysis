
import time
import random
import dask
import graphviz

from dask.distributed import Client, progress
client = Client(threads_per_worker=2, n_workers=3)
client
client = Client()
from dask.distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)

def inc(x):
    return x + 1

def double(x):
    return x * 2

def add(x, y):
    return x + y

data = [1, 2, 3, 4, 5]

output = []
for x in data:
    a = dask.delayed(inc)(x)
    b = dask.delayed(double)(x)
    c = dask.delayed(add)(a, b)
    output.append(c)

total = dask.delayed(sum)(output)
total.visualize()
total.compute()


def inc(x):
    time.sleep(random.random())
    return x + 1

def dec(x):
    time.sleep(random.random())
    return x - 1

def add(x, y):
    time.sleep(random.random())
    return x + y

%%time
x = inc(1)
y = dec(2)
z = add(x, y)
z

inc = dask.delayed(inc)
dec = dask.delayed(dec)
add = dask.delayed(add)

x = inc(1)
y = dec(2)
z = add(x, y)
z

z.visualize(rankdir='LR')
z.compute()


