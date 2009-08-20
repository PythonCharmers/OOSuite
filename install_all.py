import os
for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt']:
    print(moduleName + ' installation:')
    os.chdir(moduleName) 
    os.system('python setup.py install')
    os.chdir('..')
