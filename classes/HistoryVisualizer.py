from matplotlib import pyplot as plt


class HistoryVisualizer:
    def __chart_metric(self, history_dict, metric):
        plt.figure(figsize=(10, 5))
        plt.plot(history_dict[metric], label=f'Training {metric.capitalize()}')
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            plt.plot(history_dict[val_metric], label=f'Validation {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_history(self, history_dict):
        self.__chart_metric(history_dict, 'loss')
        self.__chart_metric(history_dict, 'accuracy')
