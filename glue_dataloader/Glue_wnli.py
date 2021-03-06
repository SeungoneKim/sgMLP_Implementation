from datasets import load_dataset, list_datasets
from torch.utils.data import Dataset

class Train_wnli(Dataset):
    def __init__(self, dataset_name, x_name_1, x_name_2, y_name, percentage=None):
        if dataset_name not in list_datasets():
            assert('Not available in Huggingface dataset')

        if percentage is None:
            train_data = load_dataset('glue', 'wnli', split='train')

        else:
            train_data = load_dataset('glue', 'wnli', split=f'train[:{percentage}%]')

        self.data_len = len(train_data)
        self.train_X_1 = train_data[x_name_1]
        self.train_X_2 = train_data[x_name_2]
        self.train_Y = train_data[y_name]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.train_X_1[index], self.train_X_2[index], self.train_Y[index]


class Val_wnli(Dataset):
    def __init__(self, dataset_name, x_name_1, x_name_2, y_name, percentage=None):
        if dataset_name not in list_datasets():
            assert('Not available in Huggingface dataset')

        if percentage is None:
            val_data = load_dataset('glue', 'wnli', split='validation')

        else:
            val_data = load_dataset('glue', 'wnli', split=f'validation[:{percentage}%]')

        self.data_len = len(val_data)
        self.val_X_1 = val_data[x_name_1]
        self.val_X_2 = val_data[x_name_2]
        self.val_Y = val_data[y_name]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.val_X_1[index], self.val_X_2[index], self.val_Y[index]


class Test_wnli(Dataset):
    def __init__(self, dataset_name, x_name_1, x_name_2, y_name, percentage=None):
        if dataset_name not in list_datasets():
            assert('Not available in Huggingface dataset')

        if percentage is None:
            test_data = load_dataset('glue', 'wnli', split='test')

        else:
            test_data = load_dataset('glue', 'wnli', split=f'test[:{percentage}%]')

        self.data_len = len(test_data)
        self.test_X_1 = test_data[x_name_1]
        self.test_X_2 = test_data[x_name_2]
        self.test_Y = test_data[y_name]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.test_X_1[index], self.test_X_2[index],self.test_Y[index]


class Total_wnli(Dataset):
    def __init__(self, dataset_name, x_name_1, x_name_2, y_name, percentage=None):
        self.train_data = Train_wnli(dataset_name, x_name_1, x_name_2, y_name, percentage)
        self.val_data = Val_wnli(dataset_name, x_name_1, x_name_2, y_name, percentage)
        self.test_data = Test_wnli(dataset_name, x_name_1, x_name_2, y_name, percentage)

    def getTrainData(self):
        return self.train_data

    def getValData(self):
        return self.val_data

    def getTestData(self):
        return self.test_data
