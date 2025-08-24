from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['train2014'],
                      test_folders=['val2014'],
                      min_size=100,
                      output_folder='./')
