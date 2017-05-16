import openml
import pandas as pd

data_sets = openml.datasets.list_datasets()
data_sets = pd.DataFrame(list(data_sets.values()))

my_query = ('NumberOfClasses==2',
            '(MinorityClassSize/MajorityClassSize) < 0.05',
            'NumberOfInstances < 1e5',
            'NumberOfInstances > 1e3',
            'status==\'active\'')

my_datasets = data_sets.query(" and ".join(my_query))
