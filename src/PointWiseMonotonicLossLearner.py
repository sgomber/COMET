import sys
import json
import pandas
import importlib
import tensorflow as tf
import numpy as np
import csv 


from datetime import datetime
from numpy.random import seed
from Envelope import generate_counter_example
from ModelCalls import generate_data
from Utils import makeDir, readConfigurations
from joblib import Parallel, delayed


CONFIG_FILE = "configurations/bostonhouseprice.txt"

class DataCollectorCallback(tf.keras.callbacks.Callback):
    def __init__(self, method, configuration, model, training_data, testing_data, train_labels_normal):
        self.method = method
        self.configuration = configuration
        self.csv_result_file = self.configuration["result_files"] + "/" + self.method + "_results.csv"
        self.result_column_names = ["Epoch#", "Training CEGs", "Training Full Loss", "Training MSE Loss"]
        self.model = model
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_labels_normal = train_labels_normal

    def on_train_begin(self, logs=None):
        with open(self.csv_result_file, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(self.result_column_names)

    def on_epoch_end(self, epoch, logs=None):
        # if not (epoch > 0 and (epoch+1) % 5 == 0):
        #     return

        layer_size = configuration["layers"]
        data_dir = configuration["weight_files"]
        for i in range(layer_size):
            weight = self.model.layers[i].get_weights()[0]
            bias = self.model.layers[i].get_weights()[1]
            np.savetxt(data_dir+"/weights_layer%d.csv"%(i),weight,delimiter=",")
            np.savetxt(data_dir+"/bias_layer%d.csv"%(i),bias,delimiter=",") 

        keys = list(logs.keys())
        # print("\n ------- CEG Analysis ------------")
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        # print(logs)
        results = get_monotonicity_counter_examples(self.configuration, self.model, self.training_data, self.testing_data)
        
        # print(results)

        counter = 0
        for res in results:
            if res[0] is not None:
                counter += 1
            if res[3] is not None:
                counter += 1

        # print("JMD: ", counter)
        
        with open(self.csv_result_file, 'a') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow([str(epoch+1), str(counter), str(logs["loss"]), str(get_mse_loss(self.model, self.training_data, self.train_labels_normal))])

        
        # print("\n ---------------------------------")

def setupFolders(configuration):
    dirname = datetime.now().strftime('%Y%m%d')+ str(datetime.now())
    dir_col_string = "Combination"
    for m in configuration['monotonic_indices']:
        dir_col_string = dir_col_string + "+" + configuration['column_names'][int(m)]
    run_data_dir = configuration['run_data_path']+ configuration['current_benchmark'] +"/" + dir_col_string +"/"+dirname +"/"
    makeDir(run_data_dir,True)
    configuration['run_data_path'] = run_data_dir

    run_data_dir = configuration['run_data_path']
    weight_files_path = run_data_dir + "folds/"
    makeDir(weight_files_path)
    configuration['weight_files'] = weight_files_path

    result_files_path = run_data_dir + "results/"
    makeDir(result_files_path)
    configuration['result_files'] = result_files_path

def get_data(configuration):
    sys.path.append('src/Models')
    make_data = importlib.__import__(configuration['model']).make_data
    train_all, train_labels_normal, test_all, test_labels_normal, min_max_dict = generate_data(make_data, configuration)

    train_all = train_all[0]
    train_labels_normal = train_labels_normal[0]
    test_all = test_all[0]
    test_labels_normal = test_labels_normal[0]

    train_labels_pwl = np.asarray([([a,b]) for a,b in enumerate(train_labels_normal)])
    test_labels_pwl = np.asarray([([a,b]) for a,b in enumerate(test_labels_normal)])

    return train_all, test_all, train_labels_normal, test_labels_normal, train_labels_pwl, test_labels_pwl, min_max_dict

def create_model_without_pwl(training_data):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(7,)))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def create_model_with_pwl(training_data):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(12,)))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    def custom_loss_envelope(model, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)      
            
        def custom_loss(y_true, y_pred):        
            
            correct_y_true = tf.gather(y_true, [1], axis=1)
            loss_PDE = tf.keras.losses.mean_squared_error(correct_y_true, y_pred)
            
            with tf.GradientTape() as t:
                t.watch(x_tensor)
                output = model(x_tensor)
            
            x_indices = tf.gather(y_true, [0], axis=1)
            x_indices = tf.reshape(x_indices, [-1])
    #         tf.print(x_indices)
            
            DyDx = t.gradient(output, x_tensor)
            # tf.print(x_tensor)
            # tf.print(DyDx)
            # tf.print(DyDx.shape)
            # tf.print(x_indices)
            # tf.print("-----------")
            
    #         temp = DyDx[x_indices]
            temp = tf.map_fn(lambda x: DyDx[int(x)], x_indices)
            # tf.print(temp)
            
            main_row = tf.gather(temp, [4], axis=1)
            main_row = tf.reshape(main_row, [-1])
    # #         tf.print(main_row)
    # #         tf.print("----JMD---")
            final = tf.map_fn(lambda x: 0.0 if x > 0.0 else -1 * x, main_row)
    #         tf.print(final)

            # tf.print(correct_y_true)
            # tf.print(y_pred)
            tf.print(loss_PDE)
            tf.print(main_row)
            tf.print(1000 * final)
            tf.print("------------------")
        
            return loss_PDE + 1000 * final
        return custom_loss

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=custom_loss_envelope(model, training_data))

    return model

def get_counter_example_u_l(configuration, data_point, monotonic_index, data_index, f_x, fold):
    counter_example_generator_upper = importlib.__import__(configuration['solver']).counter_example_generator_upper_env
    counter_example_generator_lower = importlib.__import__(configuration['solver']).counter_example_generator_lower_env

    counter_example_upper,elapsed_time_u,vio_u,ind_u = generate_counter_example(configuration, counter_example_generator_upper, data_point.copy(), monotonic_index, data_index, f_x, fold)
    counter_example_lower,elapsed_time_l,vio_l,ind_l = generate_counter_example(configuration, counter_example_generator_lower, data_point.copy(), monotonic_index, data_index, f_x, fold)

    return counter_example_upper, vio_u, ind_u, counter_example_lower, vio_l,ind_l

def get_monotonicity_counter_examples(configuration, model, training_data, testing_data):
    num_cores = int(configuration['num_cores'])
    monotonic_index = configuration['monotonic_indices']

    sys.path.append('./src/Models')
    output = importlib.__import__(configuration['model']).output

    all_results = []
    fold = 0
    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(delayed(get_counter_example_u_l)(configuration, training_data.iloc[data_index].values, monotonic_index, data_index, output(model, training_data.iloc[data_index].values)[0][0], fold) for data_index in range(0, len(training_data)))
        all_results.extend(results)

    return all_results

def get_mse_loss(model, test_all, test_labels_normal):
    x_tensor = tf.convert_to_tensor(test_all, dtype=tf.float32)
    output = model(x_tensor)
    output = tf.reshape(output, [-1]).numpy()
    
    diffs = output - test_labels_normal
    
    ans = 0
    for diff in diffs:
        ans += diff*diff
    
    return float(ans)/len(diffs)

def find_K_factor_from_training_data(configurations, train_data, train_label, threshold):
    l=["rm"]
   
    rows = []

    rows_copy = []
    labels = []

    for index, row in train_data.iterrows():
        rows.append(row.copy())
        rows_copy.append(row.copy())
    
    for val in train_label:
        labels.append(val)
        
    for r in rows:
        diction = configurations["min_max_values"]
        for k,v in diction.items():
            r[k] = float((r[k] - v[0]))/(v[1] - v[0])
        
    # print(rows)

    diction = configuration["min_max_values"]

    lis = []

    for i, r1 in enumerate(rows):
        for j, r2 in enumerate(rows): 
            fresh = 0
            if i < j:
                for k in diction.keys():
                    if k not in l:
                        fresh = fresh + (r2[k] - r1[k]) * (r2[k] - r1[k])
                
                lis.append([fresh, (i, j)])

    lis.sort()

    print(lis[:6])
    
    k_factor_sum = 0
    total_data = 0

    for data in lis:
        if data[0] > threshold:
            break
        
        p1 = rows_copy[data[1][0]]
        p2 = rows_copy[data[1][1]]
        l1 = labels[data[1][0]]
        l2 = labels[data[1][1]]

        # print(p1)
        # print(p2)
        # print(l2 - l1)
        # print("----")
        if l1 != l2:
            k_factor_sum += (float(p2[l[0]] - p1[l[0]])/(l2 - l1))
            total_data += 1
    
    print(k_factor_sum)
    print(total_data)
    
    # print(nearest, nearest_i, nearest_j)
    # print(train_data.iloc[nearest_i].values, test_data.iloc[nearest_i])
    # print(train_data.iloc[nearest_j].values, test_data.iloc[nearest_j])


if __name__ == "__main__":
    seed(4)
    tf.keras.utils.set_random_seed(2)

    configuration = readConfigurations(CONFIG_FILE)

    setupFolders(configuration)

    train_all, test_all, train_labels_normal, test_labels_normal, train_labels_pwl, test_labels_pwl, min_max_dict = get_data(configuration)
    # configuration["min_max_values"] = min_max_dict

    print(find_K_factor_from_training_data(configuration, train_all, train_labels_normal, 0.001))

    # print(configuration["min_max_values"])

    # baseline_model = create_model_without_pwl(train_all)
    # baseline_model.fit(train_all,
    #                    train_labels_normal,
    #                    epochs=15,
    #                    batch_size=32,
    #                    validation_split = 0.2,
    #                    callbacks=[DataCollectorCallback("Baseline", configuration, baseline_model, train_all, test_all, train_labels_normal)])
    # print(get_mse_loss(baseline_model, test_all, test_labels_normal))


    # pwl_model = create_model_with_pwl(train_all)
    # pwl_model.fit(train_all,
    #               train_labels_pwl,
    #               epochs=15,
    #               batch_size=32,
    #               validation_split = 0.2,
    #               callbacks=[DataCollectorCallback("PWL", configuration, pwl_model, train_all, test_all, train_labels_normal)])

    # print(get_mse_loss(pwl_model, test_all, test_labels_normal))