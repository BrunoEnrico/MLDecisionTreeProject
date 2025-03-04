import pandas as pd

class Main:
    def __init__(self):
        self.data = pd.read_csv("creditcard.csv")

    def process(self):
        print(self.data.head())


if __name__ == "__main__":
    main = Main()
    main.process()