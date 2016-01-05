def get_numpy_data(data_sframe, features, output):
    """
    :param:
     - `data_sframe`: SFrame to convert
     - `features`: list of column names
     - `output`: the target
    :return: (matrix of data_sframe columns
              with 'constant' column and 'features' column,
              array from data_sframe using 'output' column)
    """
    # add a column of 1's
    data_sframe['constant'] = 1
    
    # add the column 'constant' to the front of the features list
    # so that we can extract it along with the others
    features = ['constant'] + features
    
    # select the columns of data_sframe given by the features list
    # into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    
    # assign the column of data_sframe associated with the output
    # to the SArray output_sarray
    output_array = data_sframe[output]
    
    # the following will convert the SArray into a numpy array
    # by first converting it to a list
    output_array = output_array.to_numpy()
    return(feature_matrix, output_array)


def predict_output(feature_matrix, weights):
    """
    :param:
     - `feature_matrix`:numpy matrix containing the features
     - `weights`: vector to apply to the feature_matrix

    :return: dot-product of feature_matrix and weights
    """
    return feature_matrix.dot(weights)
