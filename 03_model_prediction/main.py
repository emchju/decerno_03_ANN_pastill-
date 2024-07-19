# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 04:55:10 2023

@author: Emelie Chandni
"""
# Import own modules
import sys
sys.path.append('../')
from main_functions import run_AI

'''
TEST_CUSTOMER_1
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000

TEST_CUSTOMER_2
Geography: Germany
Credit Score: 645
Gender: Male
Age: 44 years old
Tenure: 8 years
Balance: $ 113755.78
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: No
Estimated Salary: $ 149756.71

Exited: 1


0.0,0.0,1.0,645,1,44,8,113755.78,2,1,0,149756.71,1

So, should we say goodbye to that customer ?
'''
TEST_CUSTOMER_1 = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
TEST_CUSTOMER_2 = [[0, 0, 1, 645, 1, 44, 8, 113755, 2, 1, 0, 149756]]

customer_names = [
    'TEST_CUSTOMER_1',
    'TEST_CUSTOMER_2'
    ]

customer_data = [
    TEST_CUSTOMER_1,
    TEST_CUSTOMER_2
    ]

run_AI(customer_names, customer_data)