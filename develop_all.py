import os
for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt']:
    print(moduleName + ' in-place installation:')
    os.chdir(moduleName) 
    os.system('python setup.py develop')
    os.chdir('..')
