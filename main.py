# pip install -U sentence-transformers pandas
from sentence_transformers import SentenceTransformer, util
import time, os.path
import pandas as pd

# Set files here
files = [
    "phrases.txt", # Fill file names with phrases
]

### Start Settings
# Model you can choose from https://huggingface.co/models?library=sentence-transformers it will be download automatically
model_name = 'sentence-transformers/LaBSE'

# Minimal phrases to create cluster
min_community_size = 2

# Threshold (from -1 to 1)
threshold = 0.75

### End Settings

def get_similarity_cos(phrase1, phrase2):
    model = SentenceTransformer(model_name)
    emb1 = model.encode(phrase1, convert_to_tensor=True)
    emb2 = model.encode(phrase2, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb1, emb2)
    print(f"Cosine-Similarity: {cos_sim}")

def clustering(cluster_file=""):
    model = SentenceTransformer(model_name)

    if cluster_file == "":
        cluster_file = input("Enter file name with phrases (name.txt): ")
    out_file = cluster_file.replace(".txt", ".csv")

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(cluster_file, encoding='utf8') as fIn:
        for line in fIn.readlines():
            corpus_sentences.add(line.strip())

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    print(f"Start clustering file {cluster_file}, total lines {len(corpus_sentences)}")
    start_time = time.time()

    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)

    print("Clustering done after {:.2f} sec".format(time.time() - start_time), "total clusters: ", len(clusters))
    clustered = []  # Clustered phrases
    out = []  # Out data

    for i, cluster in enumerate(clusters):
        cluster_name = corpus_sentences[cluster[0]]
        for line in cluster:
            out.append([corpus_sentences[line], cluster_name])
            clustered.append(corpus_sentences[line])

    # Get non-clustered phrases
    not_clustered = list(set(corpus_sentences) - set(clustered))
    print(f"Clustered: {len(clustered)} clusters: {i + 1}")
    print("Not clustered: {len(not_clustered)}")

    for nc in not_clustered:
        out.append([nc, nc])

    df = pd.DataFrame(out)
    df.to_csv(out_file, header=False, index=False, sep=';')
    print(f"Success saved to file {out_file}")

if __name__ == '__main__':

    if len(phrase1) > 0 and len(phrase2) > 0:
      get_similarity_cos(phrase1, phrase2)

    # Start Clusterization
    for file in files:
      if os.path.isfile(file):
          clustering(file)
      else:
        print(f"{file} does not exists")
    # End Clusterization
