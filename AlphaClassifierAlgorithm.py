from ClassifierAlgorithm import ClassifierAlgorithm
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd


class AlphaClassifierAlgorithm(ClassifierAlgorithm):
    st = StanfordNERTagger('C:\\Python_Work\\nlp_library\\english.all.3class.distsim.crf.ser.gz',
					   'C:\\Python_Work\\nlp_library\\stanford-ner-3.9.1.jar',
					   encoding='utf-8')
    ner_classes = ['LOCATION','PERSON','ORGANIZATION']

    def train(self, input, output):
        pass

    def predict(self, input):
        output =pd.DataFrame(columns=input.columns)
        for idx, row in input.iterrows():
            label_list = []
            for text in row:
                tokenized_text = word_tokenize(text)
                classified_text = self.st.tag(tokenized_text)
                labels = self.get_predicted_labels(classified_text)
                #### Return any label as of now. We can include all for other fields like address in future. ###
                for label in labels:
                    label_list.append(label)
                    #output.loc[idx] = label
                    break
            #output = output.append(pd.Series(label_list, index=input.columns))
            output.loc[idx] = label_list
        return output

    def get_predicted_labels(self, classified_text):
        labels = {}
        for tuple in classified_text:
            if tuple[1] in self.ner_classes:
                value = tuple[1]
                if value in labels:
                    labels[value].append(tuple[0])
                else:
                    labels[value] = [tuple[0]]
            else:
                labels['NA']=['NA']
        return labels

    def serialize(self, file_path):
        pass

    def deserialize(self, file_path):
        pass