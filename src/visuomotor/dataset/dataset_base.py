

class DatasetBase:
    def default_dataset_split(self):
        raise NotImplementedError("Should be implemented in child class")
    
    def calculate_train_stats(self):
        raise NotImplementedError("Should be implemented in child class")

