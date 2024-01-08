import sys
import os

sys.path.append(os.getcwd() + '\\..\\Analysis')

from Analysis.Geometric import Geometric

g = Geometric()
g.analze()
