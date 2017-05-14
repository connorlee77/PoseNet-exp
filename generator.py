def generator(features, labels, batch_size, preprocessing_function=None):

    q = labels[1]
    x = labels[0]

    batch_features = np.zeros((batch_size, img_height, img_width, 3))
    batch_x = np.zeros((batch_size, 3))
    batch_q = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            index = np.random.choice(len(features),1)

            batch_features[i] = preprocessing_function(features[index])

            batch_x[i] = x[index]
            batch_q[i] = q[index]

        yield batch_features, {'x': batch_x, 'q': batch_q}

def directory_generator(train_dist_filepath, batch_size, preprocessing_function=None):

    data = pd.read_csv(train_dist_filepath, delim_whitespace=True, header=None, names=['path', 'X', 'Y', 'Z', 'W', 'P', 'Q', 'R'], skiprows=3)

    batch_features = np.zeros((batch_size, img_height, img_width, 3))
    batch_x = np.zeros((batch_size, 3))
    batch_q = np.zeros((batch_size, 4))

    while True:
        for i in range(batch_size):
            index = np.random.choice(len(data), 1)

            fpath = data.iloc[index]['path'].values[0]
            f = Image.open(DATA_DIR + fpath)

            f = f.resize((int(f.width * 315 / f.height), int(f.height * 315 / f.height)))

            f = np.array(f)
            if preprocessing_function:
                f = preprocessing_function(f)
            f = preprocess_input(np.float32(f))
            batch_features[i] = f

            batch_x[i] = np.float32(data.iloc[index].as_matrix(['X', 'Y', 'Z']))
            batch_q[i] = np.float32(data.iloc[index].as_matrix(['W', 'P', 'Q', 'R']))

        yield batch_features, {'x': batch_x, 'q': batch_q}