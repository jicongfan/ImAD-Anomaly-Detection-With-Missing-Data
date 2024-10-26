
from .tabular import TabularDataset

def load_dataset(dataset_name, data_path, missing_rate, split=None, mechanism='mcar'):
    """Loads the dataset."""

    implemented_datasets = ('adult', 'kdd', 'arrhythmia', 'speech', 'usoskin', 'segerstolpe', 'botnet', 'titanic', 'movielens1m', 'bladder', 'seq2_heart')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name=='adult':
        dataset = TabularDataset(root=data_path, dataset_name='Adult', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='kdd':
        dataset = TabularDataset(root=data_path, dataset_name='KDD', missing_rate=missing_rate, split=split, mechanism=mechanism)

    if dataset_name=='arrhythmia':
        dataset = TabularDataset(root=data_path, dataset_name='Arrhythmia', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='speech':
        dataset = TabularDataset(root=data_path, dataset_name='Speech', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='usoskin':
        dataset = TabularDataset(root=data_path, dataset_name='Usoskin', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='segerstolpe':
        dataset = TabularDataset(root=data_path, dataset_name='Segerstolpe', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='botnet':
        dataset = TabularDataset(root=data_path, dataset_name='Botnet', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='titanic':
        dataset = TabularDataset(root=data_path, dataset_name='Titanic', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='movielens1m':
        dataset = TabularDataset(root=data_path, dataset_name='MovieLens1M', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='bladder':
        dataset = TabularDataset(root=data_path, dataset_name='Bladder', missing_rate=missing_rate, split=split, mechanism=mechanism)
    
    if dataset_name=='seq2_heart':
        dataset = TabularDataset(root=data_path, dataset_name='Seq2_Heart', missing_rate=missing_rate, split=split, mechanism=mechanism)
    

    return dataset

