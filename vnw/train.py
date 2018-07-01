from vnw.model import ValueNetWork
from vnw.solver import ValueNetWordSolver
import utils.data_utils as data_utils


def main():
    data = data_utils.load_data(data_path="../data", split="train")
    word2idx = data["word_to_idx"]

    val_data = data_utils.load_data(data_path='../data', split='val')

    model = ValueNetWork()
    solver = ValueNetWordSolver(word2idx=word2idx, model=model, data=data, val_data=val_data, batch_size=100,
                                model_path="../model")

    solver.train()


if __name__ == '__main__':
    main()
