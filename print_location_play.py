
import sys
from contextlib import redirect_stdout

# with open('help.txt', 'w') as f:
#     with redirect_stdout(f):
#         print('it now prints to `help.text`')

original_stdout = sys.stdout # Save a reference to the original standard output

# with open('filename.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print('This message will be written to a file.')
#     sys.stdout = original_stdout

path = '/media/don/scratch/test.txt'

f = open(path, 'w')
redirect_stdout(f)
print('it now prints to `help.text`')


f = open(path, 'w')
sys.stdout = open(path, 'w')
print('test')
sys.stdout.close()

print('test 2')

f = open(path, 'w')
original_stdout = sys.stdout # Save a reference to the original standard output
sys.stdout = f # Change the standard output to the file we created.
print('first.')
print('This message will be written to a file.')
print('last')
sys.stdout = original_stdout

f = sys.stdout
f = open(path, 'w')
f = open(path, 'w', buffering=1)
print('stmt1', file=f); f.flush()
print('stmt2', file=f); f.flush()
print('stmt3', file=f); f.flush()
print('stmt4', file=f); f.flush()
print('stmt5', file=f); f.flush()
f.close()

file = open('output.out','a')
print(*args, **kwargs, file=file)

f = open(path, 'w')
f.write("hello")
f.write("close")
f.close()

print('hello')

import logging

# Create the file
# and output every level since 'DEBUG' is used
# and remove all headers in the output
# using empty format=''
logging.basicConfig(filename='output.txt', level=logging.DEBUG, format='')

logging.debug('Hi')
logging.info('Hello from AskPython')
logging.warning('exit')

