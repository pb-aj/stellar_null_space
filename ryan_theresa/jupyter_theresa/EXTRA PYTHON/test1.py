# import ruamel.yaml

# yaml = ruamel.yaml.YAML()
# data = yaml.load(open('environment.yml'))

# requirements = []
# for dep in data['dependencies']:
#     if isinstance(dep, str):
#         package, package_version, python_version = dep.split('=')
#         if python_version == '0':
#             continue
#         requirements.append(package + '==' + package_version)
#     elif isinstance(dep, dict):
#         for preq in dep.get('pip', []):
#             requirements.append(preq)

# with open('requirements.txt', 'w') as fp:
#     for requirement in requirements:
#        print(requirement, file=fp)


import numpy as np
a1 = np.ones((400, 40))

a2 = np.ones(40)

a3 = a2 * a1

print(a3.shape)

