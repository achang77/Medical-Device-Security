
import numpy as np
import pandas as pd
import os
startYear = 2007
endYear = 2019
#data_dir = './../Unique_Data/'
dataDir = '../Data/Raw_Data/'
print(os.listdir(dataDir))
destDir = '../Data/Auto_Data/'
files = []
for year in range(startYear, endYear):
    files.append('unique'+str(year)+'.xlsx')
computer_causes = {
    'Device Design',
    'Equipment maintenance',
    'Software Design Change',
    'Software Manufacturing/Software Deployment',
    'Software change control',
    'Software design',
    'Software design (manufacturing process)',
    'Software in the Use Environment'
}
not_computer_causes = {
    'Error in labeling',
    'Incorrect or no expiration date',
    'Labeling Change Control',
    'Labeling False and Misleading',
    'Labeling design',
    'Labeling mix-ups',
    'No Marketing Application',
    'Package design/selection',
    'Packaging',
    'Packaging change control',
    'Packaging process control'
}

total_c = 0
total_not_c = 0
total_und = 0
total = 0

for filename in files:
    print(filename)
    df = pd.read_excel(dataDir+filename)
    df['Fault_Class'] = ""

    for i, row in df.iterrows():
        nc = 'Not_Computer'
        c = 'Computer'
        und = 'Undetermined'
        cause = row['FDA Determined Cause']
        if cause in computer_causes:
            df.at[i, 'Fault_Class'] = c
        elif cause in not_computer_causes:
            df.at[i, 'Fault_Class'] = nc
        else:
            df.at[i, 'Fault_Class'] = und

    df = df.sort_values(by=['Fault_Class'], ascending=False)
    writer = pd.ExcelWriter(destDir + filename.split(
        '.xlsx')[0] + '_auto.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()

    df1 = df[df['Fault_Class'] != 'Undetermined']
    writer = pd.ExcelWriter(destDir + filename.split(
        '.xlsx')[0] + '_auto_determined.xlsx', engine='xlsxwriter')
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()

    c_records = len(df[df['Fault_Class'] == 'Computer'])
    not_c_records = len(df[df['Fault_Class'] == 'Not_Computer'])
    und = len(df[df['Fault_Class'] == 'Undetermined'])

    total_c += c_records
    total_not_c += not_c_records
    total_und += und
    total += c_records + not_c_records + und

    print("Computer Records: {}".format(c_records))

    print("Not Computer Records: {}".format(not_c_records))

    print("Undetermined Records: {} ".format(und))

    print("%/ Undetermined: {}".format(und / (und + c_records + not_c_records)))

print("Total Records: {}".format(total))
print("Total Computer Records: {}".format(total_c))
print("Total Not Computer Records: {}".format(total_not_c))
print("Total Undetermined Records: {} ".format(total_und))
