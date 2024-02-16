import json
import os
import math


class LaTeX:
    max_Colum_dictionary = {'FX' : 5, 'Commodities' : 9, 'Bonds' : 6, 'Equities': 6, 'Commodities_Energies' : 9, 'Commodities_Metals' : 9, 'Commodities_Softs' : 9}
    
    def __init__(self, keys, OutputPATH, Asset_class, algo, test_year, Name_dict, full_table=True):
        self.keys = list(keys)    
        self.keys += ['MACD', 'BuyAndHold']           
        self.full_table = full_table
        self.table = ''  
        self.dictionary = self._LaTeX_essentials(path='Input/LaTeX/LaTeX_keys.json')   
        self.row_names = self._LaTeX_essentials(path='Input/LaTeX/rowNames.json')     
        self.path = OutputPATH 
        self.asset_class = Asset_class    
        self.ALGO = algo     
        self.test_year = test_year       
        self.table_name = algo + "_" + Asset_class + "_" + test_year + ".tex" 
        self.output_dir = os.path.join(self.path, "LaTeX_tables") 
        self.full_names = Name_dict
        self.full_names['Portf'] = 'Equally weighted Portfolio'    
        self.full_names['BuyAndHold'] = 'Buy and Hold strategy'       
        self.full_names['MACD'] = 'MACD'   

        if self.asset_class == 'Equities':
            self.full_names['SP'] = 'E-mini S\\&P' 
            self.full_names['DX'] = '\\$-Index'                                             
        
        self.generate_latex_table()                           

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)                                   

    def _LaTeX_essentials(self, path) -> list:
        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
            return data                
        except FileNotFoundError:
            raise FileNotFoundError("Input file not found. Has been moved")                                        
    
    def generate_latex_table(self) -> None:
        caption = f' \\caption{{This table presents a comprehensive overview of {self.asset_class}\'s market performance, with signal generation produced by a {self.ALGO} agent. The assessment spans a rigorous 12-month testing period in {self.test_year}. The table further showcases the results of two essential strategies: MACD analysis and the simple Buy and Hold approach, both executed within the same timeframe.}}'

        #label = f' \\label{{Setting_{self.test_year}_{self.ALGO}_{self.asset_class}}}'        
      
        if self.asset_class not in ['Commodities', 'Commodities_Energies', 'Commodities_Metals', 'Commodities_Softs']:
            top = " \\begin{table}[hbt!] \\scriptsize \\setlength{\\tabcolsep}{2pt} \\begin{adjustbox}{center}"
            bottom = " \\end{tabular} \\end{adjustbox} INSERT_CAPTION_HERE \\end{table} "  
    
        if self.asset_class == 'Bonds':
            top = " \\begin{table}[hbt!] \\tiny \\setlength{\\tabcolsep}{0.5pt} \\begin{adjustbox}{center}"
            bottom = " \\end{tabular} \\end{adjustbox} INSERT_CAPTION_HERE \\end{table} "             

        if self.asset_class in ['Commodities', 'Commodities_Energies', 'Commodities_Metals', 'Commodities_Softs']:
            top = "\\begin{landscape} \\tiny \\begin{center} \\begin{longtable}{lccccccccc}"
            bottom = " INSERT_CAPTION_HERE \\end{longtable} \\end{center} \\end{landscape} "  
            
        bottom = bottom.replace("INSERT_CAPTION_HERE", caption)      
        #bottom = bottom.replace("LABEL", label)              
             
        row_code = ''

        def getdummy_entries(keys, offset=0) -> tuple[str, str]:
            row_code = ''
            header_code = ' & '.join([f"{self.full_names[key]}" for key in keys]) + ' \\\\\n'
            
            for row in self.row_names:
                first_key_processed = True

                for key in keys:
                    if first_key_processed:
                        if row == "Total Buy actions":
                            row_code += "\\hline "
                        row_code += row + ' & '
                        first_key_processed = False
                    else:
                        row_code += ' & '  # Add separators for the other keys

                    placeholders = row.lower().replace(' ', '') + key
                    row_code += placeholders

                if offset > 0:
                    for _ in range(offset):
                        for _ in keys:
                            row_code += '&'

                row_code = row_code.rstrip(' & ') + ' \\\\\n'

            return header_code, row_code

        num_columns = len(self.keys) + 2

        max_columns_per_row = LaTeX.max_Colum_dictionary[self.asset_class]         # ToDo Settin für jede Asset Klasse. 6 geht sich gut aus.        
        latex_rows  = []
        num_elements = len(self.keys)
        year_string = f'\\textbf{{{self.ALGO + ": " + self.test_year}}}'         

        for i in range(0, int(math.ceil(num_elements / max_columns_per_row))):
            start_index = i * max_columns_per_row
            end_index = (i + 1) * max_columns_per_row  # Calculate the end index based on the next iteration

            if end_index > num_elements:
                end_index = num_elements  # Ensure end_index doesn't exceed the number of elements

            keys = list(self.keys)[start_index:end_index]
            offset = max_columns_per_row - len(keys)

            header_code, row_code = getdummy_entries(keys, offset=offset)

            if i == 0 and self.asset_class not in ['Commodities', 'Commodities_Energies', 'Commodities_Metals', 'Commodities_Softs']:
                latex_rows.append('\\begin{tabular}{@{\extracolsep{5pt}}l' + 'c' * num_columns + '}')

            latex_row = r''' 
                \\[-1.8ex]''' + year_string + ''' & ''' + header_code + r'''
                \hline \\[-1.8ex]''' + row_code + r'''
                \hline \\[-1.8ex]'''
  
            latex_rows.append(latex_row)

            if i > 0 and i % 2 == 1 and self.asset_class in ['Commodities', 'Commodities_Energies', 'Commodities_Metals', 'Commodities_Softs']:
                latex_rows.append(r'\pagebreak')              

            if end_index == num_elements:
                break            

        if self.full_table:
            latex_table = top + '\n'.join(latex_rows) + bottom
        else:
            latex_table = '\n'.join(latex_rows) + bottom

        self.table = latex_table
        return latex_table

    def Fill_Table(self, data) -> None:
        columns_to_process = ['Portf', 'BuyAndHold', 'MACD']

        data = {key: value if value is not None else 0 for key, value in data.items()}       
        
        def mio_formatter(x):
            return f'{x / 1e6:.2f} Mio'
        
        for key in data.keys():
            for j in data[key].keys():  
                if key in columns_to_process:
                    if j in ['IC' ,'MaxVal', 'MinVal', 'abs_g', 'PV']:
                        name_in_table = self.dictionary[j] + key                        
                        self.table = self.table.replace(name_in_table, '\\textbf{' + str(mio_formatter(data[key][j]) ) + '}'  )    
                    else:                                     
                        name_in_table = self.dictionary[j] + key  
                        if data[key][j] != 0:                                              
                            self.table = self.table.replace(name_in_table, '\\textbf{' +  str(round(data[key][j],3)) + '}'  )   
                        else:                                     
                            self.table = self.table.replace(name_in_table, '\\textbf{0}'  )                                       

                else:                    
                    if j in ['IC' ,'MaxVal', 'MinVal', 'abs_g', 'PV']:
                        name_in_table = self.dictionary[j] + key                        
                        self.table = self.table.replace(name_in_table, mio_formatter(data[key][j]))   
                    else:                                     
                        name_in_table = self.dictionary[j] + key     
                        if data[key][j] != 0:                                              
                            self.table = self.table.replace(name_in_table, str(round(data[key][j],3)  ) ) 
                        else:     
                            self.table = self.table.replace(name_in_table, str(0)  )                                                  

        self.table = self.table.replace('\\textbf{0}', '')            
        print("Successfully filled")                        

    def saveTable(self) -> None:
        p = os.path.join(self.output_dir, self.table_name)        
        with open(p, "w") as file:
            file.write(self.table)

        print(f"LaTeX table has been saved to {p}")
