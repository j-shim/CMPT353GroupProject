import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib

file = sys.argv[1]

wikidata = pd.read_json(file, lines=True)

print(wikidata)