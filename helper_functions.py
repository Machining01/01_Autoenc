# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:43:16 2023

@author: Tim Reeber
"""

from pandas import read_csv as rcsv, to_timedelta
from decimal import Decimal
import re
import os
import plotly.express as px

##########################################
#####       HELPER FUNCTIONS        ######
##########################################

def extract_variables_and_units(input_string):
    # Extract the column names in the transki header and separate it from the units 
        
    # Define regular expressions to match variable names and units
    variable_pattern = r'"([^"]+)"\s*=\s*"\[([^"]+)]"'
    
    # Find all matches for variable names and units using regular expressions
    matches = re.findall(variable_pattern, input_string)
    
    # Separate variable names and units into different lists
    variable_names, units = zip(*matches)
    
    return variable_names, units

def insert_timeduration(header_info,data):
    # Add a time duration axis
    
    duration = to_timedelta(data.index * (1/header_info['Sample_Rate']), unit='s')
    data.insert(0,'time', duration)
    
    return data



##########################################
##### MAIN DATA PARSING FUNCTIONS   ######
##########################################

# Each file will be converted to a object of the class defined down below
# (ifw_data). The object holds several informations. 
# These functions get selected based on the file at hand.


def labview_read(file_path):
    # function for reading labview files
    # Assumes that one time column, one header and a standard header is present.
    header_info = {}
    df_header_linenum = 22
    header_info['Feature_index'] = float(re.findall('[0-9]+', file_path)[-1])
    
    with open(file_path, 'r') as file:
        # Read lines until the first ***End_of_Header***
        idx = 0
        for line in file:           
            value = line.strip().split('\t')
            
            if idx == df_header_linenum:
                header_info['colnames'] = (line.strip().split('\t'))
                break
                
            header_info[value[0]] = value[1:]
            idx = idx + 1
            
        # Read the rest of the file which contains data
        dec_sep = header_info['Decimal_Separator'][-1]
        delim =  header_info['Separator'][-1]
        if delim == 'Tab':
            delim = '\t'
        data = rcsv(file_path, skiprows = df_header_linenum+1, decimal = dec_sep ,delimiter = delim ,names = header_info['colnames'])
        header_info['Sample_Rate'] = 1/float(Decimal((header_info['Delta_X'][-1]).replace(',','.')))
        if data.columns.values[0] == 'X_Value':
            data.columns.values[0] = 'Time'
        # Create a time axis of type duration    
        insert_timeduration(header_info,data)
        
    return header_info, data

def qass_read(file_path):
    # Implement the function to read Qass data
    # ...
    pass

def sinedge_read(file_path):
    # Implement the function to read SinEdge data
    # ...
    pass

def transki_read(file_path):
    # Function for reading data following the TransKI-Format
    header_info = {}
    df_header_linenum = 29
    header_info['Feature_index'] = float(re.findall('[0-9]+', file_path)[-1])
    
    with open(file_path, 'r') as file:
        # Read lines until the first ***End_of_Header***
        idx = 0
        for line in file:           
            value = line.strip().split('=')
            
            if idx == df_header_linenum:
                search_ans =  re.search('{(.*)}', line.strip())
                [header_info['colnames'],header_info['Y_Unit_Label']] = extract_variables_and_units(search_ans.group(1))
                break
            header_info[value[0].replace("#","")] = value[1].replace('"','')
            idx = idx + 1
        
        if header_info['error'] == "false":
            # Read the rest of the file which contains data
            dec_sep = header_info['data_decimal'][-1]
            delim =  header_info['data_delimiter'][-1]
            if delim == 'Tab':
                delim = '\t'
            data = rcsv(file_path, skiprows = df_header_linenum+1, decimal = dec_sep ,delimiter = delim ,names = header_info['colnames'])
            if header_info['abtastrate'] != 'null':
                header_info['Sample_Rate'] = float(header_info['abtastrate'])*1000
            else:
                header_info['Sample_Rate'] = 0
            if data.columns.values[0] == 'X_Value':
                data.columns.values[0] = 'Time'
            
            # Create a time axis of type duration
            insert_timeduration(header_info,data)
        else:
            data = 0
            header_info['Sample_Rate'] = 0
            
    return header_info, data

##########################################
#####           MAIN CLASS          ######
##########################################

# class to operate on usual ifw measurement files.

class ifw_data():
    
    def __init__(self,folder,file,datatype):
        self.folder = folder
        self.file = file
        self.datatype = datatype
        
        self.fullpath = os.path.join(folder, file)
        
        if datatype == 'LabView':
            self.header, self.data = labview_read(self.fullpath)
        elif datatype == 'Qass':
            self.header, self.data = qass_read(self.fullpath)
        elif datatype == 'SinEdge':
            self.header, self.data = sinedge_read(self.fullpath)
        elif datatype == 'TransKI':
            self.header, self.data = transki_read(self.fullpath)    
        else:
            raise ValueError("Invalid datatype. Supported datatypes are 'LabView', 'Qass', 'SinEdge' and 'TransKI'.")
        
    def timeplot_per_group(self,filterstring):
        
        # presents a plotly plot of each variable group of the data in the ifw object
        
        num = int(self.header['Feature_index'])
        data = self.data.filter(like=filterstring)
        data_idx_cols = self.data.columns.get_indexer(data.columns)
        fig = px.line(data, x=self.data.time.dt.total_seconds(), 
                      y = data.columns,
                      labels={
                          "x": f"{self.data.columns[0]} [{self.header['Y_Unit_Label'][0]}]",
                          "value": f"{filterstring} [{self.header['Y_Unit_Label'][data_idx_cols[0]]}]",
                          "variable": f"Kategorie {filterstring}"},
                          title=(f'File {self.file[:-4]} | Experiment Nr.: {num}'))
        fig.show()
        

