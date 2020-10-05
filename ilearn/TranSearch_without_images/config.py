class Config:
    def __init__(self, dataset, main_path, stop_file, processed_path):
        # choose dataset name
        self.dataset = dataset

        # paths
        self.main_path = main_path
        self.stop_file = stop_file

        self.processed_path = processed_path

        self.full_path = processed_path + '{}_full.csv'.format(dataset)
        self.train_path = processed_path + '{}_train.csv'.format(dataset)
        self.test_path = processed_path + '{}_test.csv'.format(dataset)

        self.asin_sample_path = processed_path + '{}_asin_sample.json'.format(dataset)
        self.user_bought_path = processed_path + '{}_user_bought.json'.format(dataset)

        self.doc2model_path = processed_path + '{}_doc2model'.format(dataset)
        self.query_path = processed_path + '{}_query.json'.format(dataset)
        self.img_feature_path = processed_path + '{}_img_feature.npy'.format(dataset)

        self.weights_path = processed_path + 'Variable/'
        self.image_weights_path = self.weights_path + 'visual_FC.pt'
        self.text_weights_path = self.weights_path + 'textual_FC.pt'

        # embedding size
        # self.visual_size = 4096
        self.textual_size = 512
