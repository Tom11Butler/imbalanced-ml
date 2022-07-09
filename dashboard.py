"""
Author: TOM BUTLER

Python script for a streamlit dashboard showcasing the effects of imbalanced
training data on final predictions with the Titanic dataset.

"""

import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


class Dashboard:
    def __init__(self):
        self.data_scores = None
        self.data_metrics = None

        self.load_data()
        self.title()
        self.metrics()

    def title(self):
        st.title("Imbalanced Training Data")

    def load_data(self):
        # TODO load in the scores from each model
        with open("data/scores.json", "r") as file:
            self.data = json.load(file)
        self.data_metrics = pd.read_csv("data/metrics.csv")

    def sidebar(self):
        """
        Define the sidebar of options for the app.
        """
        pass

    def plot_scores(self):
        # TODO matplotlib plot of the scores for a given model
        pass

    def metrics(self):
        st.markdown("## Metrics for the different ratios")
        fig = plt.figure(figsize=(10,5))
        fig = plt.figure(figsize=(10, 5))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["accuracy"], label="Accuracy")
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["precision"], label="Precision")
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["recall"], label="Recall")
        plt.title(
            "How Standard Metrics Change With Different Class Ratios", fontdict={"fontsize": 16}
        )
        plt.legend()
        plt.show()
        st.pyplot(fig)

if __name__ == "__main__":
    dashboard = Dashboard()
