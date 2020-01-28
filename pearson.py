import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
# Read iris dataset
import xlrd
import csv
import sys

def get_csv_files(xls_file):
    csv_list=[]
    with xlrd.open_workbook(xls_file) as wb:
        sheets=wb.sheet_names()
        for sheet in sheets:
            sheet1=sheet
            sheet=wb.sheet_by_name(sheet)
            sheet1=sheet1.replace(' ','_')
            sheet_name="{}.csv".format(sheet1)
            csv_list.append(sheet_name)
            with open(sheet_name, 'w') as f:   # open('a_file.csv', 'w', newline="") for python 3
                c = csv.writer(f)
                for r in range(sheet.nrows):
                    #print(sheet.row_values(r))
                    c.writerow(sheet.row_values(r))
    return csv_list

csv_list=get_csv_files(sys.argv[1])

for csv_file in csv_list:
    dataset=pd.read_csv(csv_file)
    # Print attributes name
    print(dataset.columns.values)
    # Drop last column from dataset
    df=dataset.drop('Dates',1)

    # Generate pearson correlation matrix
    cor=df.corr(method='pearson')
    print (cor)

    # Printing correlation in heat matrix
    cm=plt.cm.viridis
    sns.heatmap(cor,cmap=cm,linewidths=0.1,linecolor='white',annot=True)
    plt.title(csv_file[:-4])
    plt.show()
