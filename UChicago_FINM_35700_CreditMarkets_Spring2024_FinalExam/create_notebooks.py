import nbformat as nbf
import json

# Create Jupyter notebook
nb = nbf.v4.new_notebook()

cells = []

# Title cell
title_md = '''# Credit Markets Final Exam - Complete Solution
## FINM 35700 - Spring 2024
### UChicago Financial Mathematics

**Date:** May 3, 2024

This notebook contains complete solutions to all 6 problems in the Credit Markets final exam.

**Problems:**
1. Overall Understanding of Credit Models (40 points)
2. Risk and Scenario Analysis for AAPL Bond (20 points)
3. CDS Calibration and Pricing (20 points)
4. Derivation of Bond PVs and DV01s in sympy (25 points)
5. LQD ETF Basket Analysis - Bucketed DV01 Risks (25 points)
6. Nelson-Siegel Model for ORCL Curve (25 points)

**Total:** 155 points
'''
cells.append(nbf.v4.new_markdown_cell(title_md))

# Setup cell
setup_code = '''# Import required libraries
import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
import sympy as sp
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plot display
%matplotlib inline

# Import credit market tools
import sys
sys.path.append('/home/user/Credit-Markets/UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam')
from credit_market_tools import *

# Set calculation date
calc_date = ql.Date(3, 5, 2024)
ql.Settings.instance().evaluationDate = calc_date
as_of_date = pd.to_datetime('2024-05-03')

# Data path
data_path = '/home/user/Credit-Markets/UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/'

print(f'Calculation Date: {calc_date}')
print(f'All packages loaded successfully!')
'''
cells.append(nbf.v4.new_code_cell(setup_code))

# Read the full solution
with open('/home/user/Credit-Markets/UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/Credit_Markets_Final_Exam_Complete_Solution.py', 'r') as f:
    full_solution = f.read()

# Split solution by problems
problem_splits = [
    ('PROBLEM 1: Overall Understanding of Credit Models', 'PROBLEM 2'),
    ('PROBLEM 2: Risk and Scenario Analysis for AAPL Bond', 'PROBLEM 3'),
    ('PROBLEM 3: CDS Calibration and Pricing', 'PROBLEM 4'),
    ('PROBLEM 4: Derivation of Fixed Rate Bond PVs and DV01s in sympy', 'PROBLEM 5'),
    ('PROBLEM 5: LQD ETF Basket Analysis', 'PROBLEM 6'),
    ('PROBLEM 6: Nelson-Siegel Model for ORCL Curve', 'SUMMARY'),
]

for i, (start_marker, end_marker) in enumerate(problem_splits):
    # Find problem code
    start_idx = full_solution.find(start_marker)
    end_idx = full_solution.find(end_marker)

    if start_idx != -1:
        if end_idx == -1:
            problem_code = full_solution[start_idx:]
        else:
            problem_code = full_solution[start_idx:end_idx]

        # Extract the print statements and code
        lines = problem_code.split('\n')
        code_lines = [l for l in lines if not l.startswith('###') and not l.strip().startswith('print("="')]
        clean_code = '\n'.join(code_lines)

        # Add markdown header
        cells.append(nbf.v4.new_markdown_cell(f'# Problem {i+1}'))

        # Add code cell
        cells.append(nbf.v4.new_code_cell(clean_code))

nb['cells'] = cells

# Save Jupyter notebook
with open('/home/user/Credit-Markets/UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/Credit_Markets_Final_Exam_Solution.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Jupyter notebook created successfully!')
print(f'Total cells: {len(cells)}')
