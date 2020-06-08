"""
    This module encapsulates logic of data preprocessing
    It partialy based on source code from a repository:
    https://github.com/neuralmind-ai/electricity-theft-detection-with-self-attention
"""
import os
import pandas
import numpy
from sklearn.preprocessing import quantile_transform
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """ Represents fraud dataset """
    def __init__(self, x, y):
        """ Initialization & NaN values transformation (NaN => 0 + bit mask """
        self.x = x
        self.xnan = x.copy()
        self.y = y
        self.x[numpy.isnan(self.x)] = 0.
        self.xnan[~numpy.isnan(self.xnan)] = 0.
        self.xnan[numpy.isnan(self.xnan)]  = 1.
        self.x = numpy.stack((self.x, self.xnan), axis=1)

    def __len__(self):
        """ Returns length of dataset """
        return len(self.x)

    def __getitem__(self, index):
        """ row's getter """
        x, y = self.x[index], self.y[index]
        return x.astype(numpy.float32), y.astype(numpy.int64)
   

class FraudData:
    """
        Represents electricity data and various metrics
    """
    def __init__(self, filepath='data.csv'):
        """ Initializes internal variables """
        self.__csv_data = None
        self.__raw_data = None
        self.__normalized_data = None
        self.__filepath = filepath
        self.__thief_ac = None
        self.__thief_norm_ac = None
        self.__regular_ac = None
        self.__regular_norm_ac = None
        self.__dataset = None
    
    def __load_raw_data(self):
        """ Loads and transform raw data """
        self.__csv_data = pandas.read_csv(self.__filepath, index_col=0)
        self.__raw_data = self.__csv_data.copy()
        self.__raw_data.drop(['FLAG'], axis=1, inplace=True)
        self.__raw_data = self.__raw_data.T.copy()
        self.__raw_data.index = pandas.to_datetime(self.__raw_data.index)
        self.__raw_data.sort_index(inplace=True, axis=0)
        self.__raw_data = self.__raw_data.T
        self.__raw_data['FLAG'] = self.__csv_data.FLAG
    
    @property
    def raw(self):
        """ Getter for raw dataframe """
        if self.__raw_data is None:
            self.__load_raw_data()
        return self.__raw_data
        
    @property
    def normalized(self):
        """ Getter for normalized dataframe """
        if self.__normalized_data is None:
            self.__normalize(self.__filepath)
        return self.__normalized_data
    
    @property
    def thief_ac(self):
        """ Thief's autocorrelation getter """
        if self.__thief_ac is None:
            self.__thief_ac = calculate_autocorrelation(self.raw[self.raw['FLAG'] > 0.0].drop(
                'FLAG', axis=1))
        return self.__thief_ac
    
    @property
    def norm_thief_ac(self):
        """ Thief's autocorelation getter based on normalized dataset """
        if self.__thief_norm_ac is None:
            self.__thief_norm_ac = calculate_autocorrelation(
                self.normalized[self.normalized['flags'] > 0.0].drop('flags', axis=1))
        return self.__thief_norm_ac
    
    @property
    def regular_ac(self):
        """ Regular client's autocorrelation getter """
        if self.__regular_ac is None:
            self.__regular_ac = calculate_autocorrelation(self.raw[self.raw['FLAG'] < 1.0].drop(
                'FLAG', axis=1))
        return self.__regular_ac
    
    @property
    def norm_regular_ac(self):
        """ Regular client's autocorelation getter based on normalized dataset """
        if self.__regular_norm_ac is None:
            self.__regular_norm_ac = calculate_autocorrelation(
                self.normalized[self.normalized['flags'] < 1.0].drop('flags', axis=1))
        return self.__regular_norm_ac
    
    @property
    def dataset(self):
        """ Tourch dataset getter """
        if self.__dataset is None:
            self.__dataset = FraudDataset(x=self.raw.drop('FLAG', axis=1), 
                                          y=self.raw['FLAG'])
        return self.__dataset
        
    def __normalize(self, filepath):
        """ Normalizes raw data and makes new normalized dataframe """
        self.__normalized_data = self.raw.drop(['FLAG'], axis=1)
        quantile = quantile_transform(self.__normalized_data.values, n_quantiles=10, 
                random_state=0, copy=True, output_distribution='uniform')
        self.__normalized_data = pandas.DataFrame(data=quantile,
                columns=self.__normalized_data.columns, index=self.__normalized_data.index)
        
        self.__normalized_data['flags'] = self.__raw_data['FLAG']
        # TODO: make NULL padding for week days instead of droping a partial-defined weeks
        self.__normalized_data = self.__normalized_data.iloc[:, 5:] # aligns

        
class CNNDataset(FraudDataset):
    """ View of FraudDataset as 2D image """
    def __getitem__(self, index):
        """ item's getter """
        # TODO: make precalculation 
        x, y = self.x[index], self.y[index]
        x = x.reshape(2,-1,7)
        return x.astype(np.float32), y.astype(np.int64)        


def calculate_autocorrelation(frame):
    """ Calculates autocorelation for input dataframe that has rows as series and columns as timestamps """
    # thief = data.raw[data.raw['FLAG']==1.0].drop('FLAG', axis=1)
    result = pandas.DataFrame(dtype=numpy.float64)
    row_counter = 1
    skipped = 0
    for key, row in frame.iterrows():
        start = row.index[((row.shift(1).isnull())) & (~row.isnull())]
        end = row.index[(row.shift(-1).isnull()) & (~row.isnull())]
        assert start.shape == end.shape
        series = [row.loc[start.tolist()[i]:end.tolist()[i]] 
                  for i in range(0,start.size)]
        part_counter = 0
        sizes = [ser.size for ser in series]
        if len(sizes)==0:
            # Skips rows without values
            skipped = skipped + 1
            continue
        assert len(sizes) != 0
        corr_interval = 30
        for serie in series:
            if serie.shape[0] < corr_interval*2:
                # skips too small parts
                skipped = skipped + 1
                continue
            result['C{i}_P{j}'.format(i=row_counter, j=part_counter)] = pandas.Series(
                [serie.autocorr(lag=lag) for lag in range(0,corr_interval)], dtype=float)
            part_counter = part_counter + 1
        row_counter = row_counter + 1
    
    return result
    
def download_data():
    # Download data from GitHub repository
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip')
    # Unzip downloaded data
    os.system('cat data.z01 data.z02 data.zip > data_compress.zip')
    os.system('unzip -n -q data_compress')