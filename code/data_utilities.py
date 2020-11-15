import matplotlib.pyplot as plt
import seaborn as sns

class DataWrapper(object):

    def __init__(self):
        import pickle
        with open('../data/train_test_split.pkl', mode='rb') as f:
            train_test_split = pickle.load(f)
        self.X_train = train_test_split['X_train']
        self.y_train = train_test_split['y_train']
        self.X_test = train_test_split['X_test']
        self.y_test = train_test_split['y_test']


class DataVisualization(DataWrapper):

    def __init__(self):
        super().__init__()

    def barplot_waterpoints(self, feature_list):
        for col in feature_list:
            fig, ax = plt.subplots()
            fig.set_figheight(8)
            fig.set_figwidth(10)
            ax.set_title(f'Distribution of Waterpoints by {col}')
            sns.countplot(y=col, data=self.X_train,
                          order=self.X_train[col].value_counts().index, ax=ax);
            fig.savefig(f'../images/waterpoints_by_{col}.png', bbox_inches='tight')
        return fig
