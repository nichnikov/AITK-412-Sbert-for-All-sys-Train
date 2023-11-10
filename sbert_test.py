import os
import json
import pandas as pd
# from src.texts_processing import TextsTokenizer
# from src.config import (logger, PROJECT_ROOT_DIR)
# from src.classifiers import FastAnswerClassifier
from sentence_transformers import (SentenceTransformer, 
                                   util)


model = SentenceTransformer(os.path.join(os.getcwd(), "models", "all_sys.transformers"))
quries_df = pd.read_csv(os.path.join(os.getcwd(), "data", "val_dataset.csv"), sep="\t")

quries_dicts = quries_df.to_dict(orient="records")
print(quries_dicts[:5])

test_results = []
for num, d in enumerate(quries_dicts):
    print(num, "/", len(quries_dicts))
    emb1 = model.encode(d["query1"])
    emb2 = model.encode(d["query2"])
    score = util.cos_sim(emb1, emb2)
    test_results.append({"query1": d["query1"], 
                         "query2": d["query2"], 
                         "predict": score.item(), 
                         "true": d["score"]})

test_results_df = pd.DataFrame(test_results)
print(test_results_df)

test_results_df.to_csv(os.path.join(os.getcwd(), "results", "test_results.csv"), sep="\t", index=False)

'''
embs1 = model.encode([d["query1"] for d in quries_dicts[:100]])
embs2 = model.encode([d["query2"] for d in quries_dicts[:100]])
predict_scores = util.cos_sim(embs1, embs2)
true_scores = [d["score"] for d in quries_dicts[:100]]

print(predict_scores.shape)'''