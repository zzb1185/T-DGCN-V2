import input_data as id


def load_data(data_name):
    # if data_name == 'test_tdgcn':
    #     data, adj = id.load_testtdgcn_data('test')
    # if data_name == 'test_tgcn':
    #     data, adj = id.load_testtgcn_data('test')

    if data_name == 'test':
        data, spametrix = id.load_test_data('test')

    return data, spametrix
