# %%
# %%
# with tf.Session() as session:
#     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     message_embeddings = session.run(embed(messages))
#     for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#         print("Message: {}".format(messages[i]))
#         print("Embedding size: {}".format(len(message_embedding)))
#         message_embedding_snippet = ", ".join(
#             (str(x) for x in message_embedding[:3]))
#         print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

# %%
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
# %%
# @return a list of dataframes


def getData(dataDir='/Users/adamchang/programming/Medical Device Security/Embedding/Data/Auto_Data/',
            fileSuffix='.xlsx', startYear=2014, endYear=2019,
            filenamePattern='unique[0123456789]{4}_auto_determined.xlsx'):
    files = os.listdir(dataDir)
    dfList = []
    for filename in files:
        if re.match(filenamePattern, filename) is not None:
            year = int(filename.split('.')[0].split('unique')[1].split('_')[0])
            if year >= startYear and year <= endYear:
                name = dataDir + filename.split('.')[0] + fileSuffix
                df = pd.read_excel(name)
                dfList.append((df, filename.split('.')[0]))
    return dfList


def labelToOrdinal(s):
    if str(s) == 'Not_Computer':
        return 0
    else:
        return 1


def ordinalToLabel(d):
    if (d == 0).bool():
        return 'Not_Computer'
    else:
        return 'Computer'


def train(xTrain, yTrain):
    # model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
    #                       random_state=42, max_iter=50, tol=1e-3)
    model = LogisticRegression(random_state=0)
    model.fit(xTrain, yTrain)
    # pickle.dump(model, open(
    #     '/Users/adamchang/programming/Medical Device Security/Embedding/Data/Aux_Data/trained_model_embedding.txt', 'w'))
    # print('model dumped')
    return model


def test(testData, model, session):
    for dfTestTuple in testData:
        dfTest = dfTestTuple[0]
        messages = dfTest['Reason for Recall'].values.astype('U')
        message_embeddings = session.run(embed(messages))
        xTest = message_embeddings
        yPred = model.predict(xTest)
        yLabel = dfTest['Fault_Class'].apply(labelToOrdinal).values

        print('performance on ' + dfTestTuple[1])
        print('percent computer related predicted')
        print(np.sum(yPred) / float(np.size(yPred)))
        print('percent computer related true')
        print(np.sum(yLabel) / float(np.size(yLabel)))
        print(classification_report(yLabel, yPred))
        print("accuracy: {:.2f}%".format(accuracy_score(yLabel, yPred) * 100))
        print("f1 score: {:.2f}%".format(f1_score(yLabel, yPred) * 100))

        dfPred = pd.DataFrame(
            yPred, columns=['Predicted Fault Class']).apply(ordinalToLabel,
                                                            axis=1)
        dfTest = pd.concat([dfTest, dfPred],
                           axis=1)
        filename = '/Users/adamchang/programming/Medical Device Security/Embedding/Data/Auto_Data/' + \
            dfTestTuple[1] + '_prediction.xlsx'
        dfTest.to_excel(filename, index=False, engine='xlsxwriter')

# %%
# os.environ['TFHUB_CACHE_DIR'] = '../tf_cache/'
# @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])

# %%
df = pd.read_excel(
    '/Users/adamchang/programming/Medical Device Security/Embedding/Data/Merged_Data/Merged_Final_Unique_Recalls_2007_2013_groundtruth.xlsx')
messages = df['Reason for Recall'].values
xTrain = session.run(embed(messages))
# %%
yTrain = df['Fault Class']
yTrain = yTrain.apply(labelToOrdinal)
model = train(xTrain, yTrain)
# %%
testData = getData()
# %%
test(testData, model, session)


# %%
