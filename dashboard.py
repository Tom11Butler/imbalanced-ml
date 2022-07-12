"""
Author: TOM BUTLER

Python script for a streamlit dashboard showcasing the effects of imbalanced
training data on final predictions with the Titanic dataset.

"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
st.set_page_config(layout="wide")


def highlight(df):
    if df['true'] == 1:
        return ['background-color: lightgreen'] * len(df)
    else:
        return ['background-color: white'] * len(df)


class Dashboard:
    def __init__(self):
        self.data_scores = None
        self.data_metrics = None
        self.data_true = None
        self.ratio = 1

        self.load_data()
        self.title()
        self.sidebar()
        self.plot_scores()
        # self.metrics()

    def title(self):
        st.title("Imbalanced Training Data")

    def load_data(self):
        # TODO load in the scores from each model
        with open("data/scores.json", "r") as file:
            self.data_scores = json.load(file)
        self.data_metrics = pd.read_csv("data/metrics.csv")
        self.data_metrics['model_ratio'] = pd.to_numeric(self.data_metrics['model_ratio'])
        self.data_true = pd.read_csv("data/true.csv")


    def sidebar(self):
        """
        Define the sidebar of options for the app.
        """

        step = np.around(self.data_metrics['model_ratio'].diff()[1], 1)

        self.ratio = st.slider(
            label="Choose the class ratio to inspect",
            min_value=self.data_metrics['model_ratio'].min(),
            max_value=self.data_metrics['model_ratio'].max(),
            value=1.0,
            step=step
        )

    def plot_scores(self):
        # TODO matplotlib plot of the scores for a given model

        col1, col2 = st.columns([3,1])

        df_scores = pd.DataFrame(self.data_scores[f'{self.ratio}'], columns=['proba'])
        df_scores = pd.merge(df_scores, self.data_true, left_index=True, right_index=True)

        fig = plt.figure(figsize=(20, 5))
        plt.title(
            f"Histogram of scores for a training class ratio of {self.ratio}",
            fontdict={"fontsize": 16}
        )
        sns.histplot(
            df_scores.query("true==0")["proba"],
            bins=[0.05 * x for x in range(21)],
            label="Died",
        )
        sns.histplot(
            df_scores.query("true==1")["proba"],
            bins=[0.05 * x for x in range(21)],
            label="Survived",
            color='r',
            alpha=0.7
        )
        plt.legend()

        df_scores = df_scores.sort_values("proba", ascending=False).reset_index(drop=True)
        # df_scores = df_scores.style.apply(highlight, axis=1)

        with col1:
            st.pyplot(fig)
        with col2:
            st.dataframe(df_scores.style.apply(highlight, axis=1))

    def metrics(self):
        st.markdown("## Metrics for the different ratios")
        fig = plt.figure(figsize=(20, 5))
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["accuracy"], label="Accuracy")
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["precision"], label="Precision")
        plt.plot(self.data_metrics["model_ratio"], self.data_metrics["recall"], label="Recall")
        plt.title(
            "How Standard Metrics Change With Different Class Ratios", fontdict={"fontsize": 16}
        )
        plt.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    dashboard = Dashboard()
