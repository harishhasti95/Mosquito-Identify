import pandas as pd
import os
import shutil

def data_prep_train():
    train = pd.read_csv('train.csv')
    class_labels = list(train['class_label'].unique())
    for each_class_label in class_labels:
        temp_directory_to_create = './train/' + each_class_label

        if not os.path.exists(temp_directory_to_create):
            os.makedirs(temp_directory_to_create)

    for index, df_row in train.iterrows():
        temp_path_dest = './train/' + df_row['class_label'] + '/' + df_row['img_fName']
        temp_path_src = './train_images/' + df_row['img_fName']
        shutil.copyfile(temp_path_src, temp_path_dest)

data_prep_train()