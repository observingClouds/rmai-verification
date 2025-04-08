class Renamer():
    def __init__(self,rename_dict):
        self.rename_dict = rename_dict
    
    def execute(self,input_ds):
        new_dict = dict()
        for new_name,old_names in self.rename_dict.items():
            for name in input_ds.keys():
                if name in old_names:
                    new_dict[name]=new_name
                else:
                    pass
        new_ds = input_ds.rename(new_dict)
        return new_ds

