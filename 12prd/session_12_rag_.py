"""
 rag example: MongoDB + Gemma
 https://huggingface.co/learn/cookbook/en/rag_with_hugging_face_gemma_mongodb
"""
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import ollama
from torch.hub import download_url_to_file

model_embeddings = BGEM3FlagModel(
    'BAAI/bge-m3',  # multilingual model from HF
    use_fp16=True,  # shorter embeddings
    return_sparse=False,
    return_colbert_vecs=False,
    # not just 1 vector for each sentence, but a matrix where each line represents different similarity features
    return_dense=True
)


def path_workaround():
    ## why this bug hapened!?
    curdir = os.getcwd()
    print(f"curdir {curdir}")
    rootpath = 'data'
    if not curdir.endswith('PyCharmMiscProject'):
        rootpath = '../' + rootpath
    path_dataset = f'{rootpath}/datasets'

    print(f"{path_dataset} exists: {os.path.exists(path_dataset)}")
    return rootpath, path_dataset


root, dataset_path = path_workaround()

os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
path_dataset = dataset_path + "/movies_info.csv"
if not os.path.exists(path_dataset):
    # https://www.kaggle.com/datasets/rushildhingra25/movies-info?select=movies_info.csv
    download_url_to_file(
        "https://share.yellowrobot.xyz/quick/2025-8-1-F9927D15-6D09-44AD-89F5-169A01A8C6CF.csv",
        path_dataset,
        progress=True
    )
df_movies = pd.read_csv(path_dataset)
# columns: original_title, overview, genres (["drama", "adventures",...])

embs_dense_overviews = []
path_embeddings = dataset_path + "/movies_embeddings.pkl"
if not os.path.exists(path_embeddings):
    download_url_to_file(
        "https://share.yellowrobot.xyz/quick/2025-12-10-1B3250CF-D5B8-4E9B-969B-098832E60DB7.pkl",
        path_embeddings,
        progress=True
    )
if os.path.exists(path_embeddings):
    with open(path_embeddings, "rb") as f:
        embs_dense_overviews = pickle.load(f)
else:  # if prepared movies embeddings failed, then prepare them yourself
    batch_size = 10
    overview_texts = df_movies["overview"].values
    idx = 0
    for overview_text in tqdm(overview_texts, desc="Encoding movie overviews"):
        print(idx, overview_text)
        idx += 1
        dense_vecs = np.zeros(1024, dtype=np.float16)
        embedding = model_embeddings.encode(overview_text)
        if embedding['dense_vecs'] is not None:
            dense_vecs = embedding['dense_vecs']
        embs_dense_overviews.append(dense_vecs)
    with open(path_embeddings, "wb") as f:
        pickle.dump(embs_dense_overviews, f)
embs_dense_overviews = np.array(embs_dense_overviews)

# TODO
""" 
1) Add an option for users to specify which movie types and plot elements they dislike. 
Use negative prompting in classification task (for extra safety)
  or/and disjoint selection of most similar movies that matches disliked movies. (retrieval)
  Ex: I dislike horror and slasher movies
    I don’t like movies where the plot revolves around time travel.
    I dislike slow-paced, artsy, or overly philosophical films.
    Avoid movies focused on politics, elections, or government conspiracies
    Avoid movies with excessive violence, gore, or torture scenes
    I dislike stories involving love triangles, cheating, or toxic relationships.
  
  > In other words:
    Find movies similar to what the user dislikes
    Then exclude those movies (or their neighbors) from the final selection pool
    (Provide only safe candidates)

2) Add function to classify Genre from first input movie_desc using LLM zero or few-shot methods 
 and then choose only as candidates generes that match movie description
   - classification with genere enum as answer <- extract genres available
     (primary_genre, secondary_genres)
   - filter candidates:
        candidates = [
            movie for movie in all_movies
            if movie.genre in allowed_genres
        ]

3) Add specifically designed re-ranker instead of prompt based classifier for choice of best fitting answer 
https://huggingface.co/Qwen/Qwen3-Reranker-8B
    A re-ranker model is not a general chat model and not a classifier in the prompt sense.
    Instead of asking: “Which movie fits best?”
    You ask the model to score (query, candidate) pairs. 
    Then you sort by score and pick the top result.

Alternative task: Use Kaggle Recipes Dataset : 64k Dishes to create recepie recommendation system 
- system should ask what user likes to eat, what user does not like to eat and available ingredients available 
  - free text prompts. https://www.kaggle.com/datasets/prashantsingh001/recipes-dataset-64k-dishes
"""

""" top up model
  Retrieval [context]
    [All movies]
    ↓
    Genre classification (LLM)
    ↓
    Disliked-movie filtering
    ↓
    Candidate retrieval (top-K)
    ↓
    Re-ranker (Qwen3-Reranker-8B)
    ↓
    Best movie
    
  Generation
"""
# https://www.linkedin.com/pulse/structured-output-gemma3-ali-afshar-nadae/
# provide the exact response format model should return
class Response(BaseModel):
    chain_of_thought: str = Field(  # adding this to not thinking model increases accuracy in avg up to 30%
        ...,
        description='Explain step by step how you compared these movies and why some movies would fit better.'
    ),
    best_fitting_movie: Literal['1', '2', '3', '4', '5'] = Field(  # limits to specified values
        ...,  # reusing existing field in parent
        description="ID number of most similar movie plot. Answer only with 1,2,3,4,5."  #
    )


"""
 I. Rietrieval
"""
# movie_desc=input("Describe the movie you wish to watch")
movie_desc = "sci-fi movie about smth cool"
""" TODO does used embeder do that?
Before generating embeddings, I applied several preprocessing techniques to clean and normalize the text. These included:
    Lowercasing
    Removing punctuation
    Tokenization
    Stopword removal
    Lemmatization or stemming
These steps reduce noise and help ensure the embeddings capture the semantic meaning more effectively.
https://journals-times.com/2025/06/25/rag-retrieval-augmented-generation-and-embedding-part-1/
"""
movie_desc_emb = model_embeddings.encode(movie_desc)
# {dense_vecs, sparse_vecs,colbert_vecs
movie_desc_emb_dense = movie_desc_emb['dense_vecs']

cos_sim = embs_dense_overviews @ movie_desc_emb_dense  # simplified version of A.B/(||A||*||B||), where ||x|| is the vector length

closest_idxs = np.argsort(cos_sim)[-5:]  # select x closest entries; closest has hotter value
# (by default values are sorted ascending, so take from theend
print(closest_idxs)
print("===" * 10)

movie_plots = df_movies.iloc[closest_idxs].values  # .iloc = integer-location based indexing
# .values: Converts the pandas Series into a NumPy array
#           Index labels and column names are removed
# print(movie_plots)

movie_plot_context = ""
for i, plot in enumerate(movie_plots):
    movie_plot_context += f"<movie>{i + 1}. {plot}<movie>\n"

print(f">>> Movie plot context: {movie_plot_context}")
print("<<<" * 10)
"""
 2. Augmentation
  - if you want/need to control the sources of answer, then always provide the facts to base answer on.
    - those will have higher similarity score by default
    - it should hallucinate less on the referenced data
  - positional embedding and newer cyclic embeddings has the same problem:
     - the first word of the sentence have the greatest importance than the others (~10x).
     - models trained on specific language performs better when queried in that specific language
       - dataset is not homogeneous so if the query starts with more frequent keywords, the result may be better
       https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-generation-process
  - multi turn sessions (with a lot of follow up questions) impacts the quality of results
     In such cases, users might start with an underspecified instruction and further clarify their needs through turn interactions.
     Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: 
       - a minor loss in aptitude and a significant increase in unreliability. 
         We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, 
          on which they overly rely. 
         In simpler terms, we discover that *when LLMs take a wrong turn in a conversation, they get lost and do not recover*. 
         - models that achieve stellar (90%+) performance in the lab-like setting of fully-specified, single-turn conversation 
         struggle on the exact same tasks in a more realistic setting when the conversation is underspecified and multi-turn.
         !!! LLM systems are typically evaluated in single-turn, fully-specified settings !!!
       https://arxiv.org/abs/2505.06120 
  - long context degrades quality
    Performance Drops with Length: Models lose accuracy as input token count increases, with drops of 20–50% from 10k to 100k+ tokens in NIAH tasks.
        Low-similarity queries (requiring semantic reasoning) degrade faster.
    Distractors Hurt More: Adding related but irrelevant info (distractors) amplifies errors. 
        Claude models abstain conservatively; GPTs hallucinate confidently.
    Context Structure Matters: Surprisingly, shuffled, incoherent contexts outperform logically structured ones, 
        suggesting models struggle with structured attention.
    Output Scaling Fails: When outputs scale with inputs (e.g., replicating long sequences), errors spike, 
        with refusals up to 4% and misplacements common.

    https://github.com/chroma-core/context-rot
    https://research.trychroma.com/context-rot
    https://medium.com/@animesh1997/context-rot-is-a-big-problem-and-graphs-are-the-promising-fix-for-coding-agents-30be152c49c6

  - applying personas is not always helpful (yet the test is biased and the results are questionable) see https://arxiv.org/abs/2308.07702
    - if adding persona add thorough context to act upon
    - apply it in 2 turns (introduce - approval response - query)
  https://github.com/Jiaxin-Pei/Prompting-with-Social-Roles
  https://www.prompthub.us/blog/role-prompting-does-adding-personas-to-your-prompts-really-make-a-difference
  https://arxiv.org/pdf/2311.10054
"""
# ollama.pull('gemma3:2b')
"""
    messages = [ 
      #Decision policy + constraints
        SYSTEM_MESSAGE, <- introduce system prompt with main instructions, including negative prompting 
      #Behavioral anchoring
        SHOT_1_USER,
        SHOT_1_ASSISTANT,
      #Instance-specific data
        FINAL_USER_QUERY
    ]

"""
response = ollama.chat(
    model='gemma3:1b',
    messages=[
        {
            'role': 'user',
            'content': f"choose movie which fit description. \n"
                       # +'output movie number: 1,2,3,4,5.\n'
                       + f'<description> {movie_desc} </description>\n'
                       + movie_plot_context
        },
    ]
    , options={'temperature': 0}
    , format=Response.model_json_schema() # Syntax + structure enforcement
    , logprobs=True
    , top_logprobs=5
)
"""
  #Multi shot prompting
   - We presented a 175 billion parameter language model which shows strong performance on many NLP tasks and
      benchmarks in the zero-shot, one-shot, and few-shot settings, in some cases nearly matching the performance of
      state-of-the-art fine-tuned systems, as well as generating high-quality samples and strong qualitative performance at
      tasks defined on-the-fly.
    https://arxiv.org/pdf/2005.14165

    In this paper, we show that ground truth demonstrations are in fact not required for effective incontext learning (Section 4). 
    Specifically, replacing the labels in demonstrations with random labels barely hurts performance in a range 
     of classification and multi-choice tasks (Figure 1). The result is consistent over 12 different models including the
     GPT-3 family - This strongly suggests, counter-intuitively, that the model does not rely on the input-label mapping 
     in the demonstrations to perform the task.
    We find that: 
      (1) the label space and the distribution of the input text specified by the demonstrations 
        are both key to in-context learning (regardless of whether the labels are correct for individual inputs); 
      (2) specifying the overall format is also crucial, e.g., when the label space is unknown, using random English words as labels 
        is significantly better than using no labels; and
      (3) meta-training with an in-context learning objective (Min et al., 2021b) magnifies these effects—the models almost 
        exclusively exploit simpler aspects of the demonstrations like the format rather than the input-label mapping.

    Figure 8 shows that using out-of distribution inputs instead of the inputs from the training data 
      significantly drops the performance ... both in classification and multichoice, by 3–16% in absolute. 
    Based on Figure 10, removing the format is close to or worse than no demonstrations, indicating the importance of the format.

    learning a new task can be interpreted more broadly: 
        it may include adapting to specific input and label distributions and the format suggested by the demonstrations, 
        and ultimately getting to make a prediction more accurately. With this definition of learning, the model does learn
        the task from the demonstrations. Our experiments indicate that the model does make use of aspects of the demonstrations 
    https://arxiv.org/pdf/2202.12837

     Our theoretical study shows that larger language models are easily overfitted to input
        noise and label noise during in-context learning, while smaller models are robust to noise, leading to
        different behaviors. Our empirical results support our claim and are consistent with previous work.
     https://openreview.net/pdf?id=2J8xnFLMgF

     One strategy worth testing is placing your most critical example last in the order. 
     LLMs have been known to place significant weight on the last piece of information they process.
     When evaluating DeepSeek-R1, we observe that it is sensitive to prompts. 
     Few-shot prompting consistently degrades its performance.
      Therefore, we recommend users directly describe the problem and specify the output format using a zero-shot setting
       for optimal results.

    OpenAI's guidance: “Limit additional context in retrieval-augmented generation (RAG):
       When providing additional context or documents, include only the most relevant information to prevent the model 
        from overcomplicating its response.”
     https://www.prompthub.us/blog/the-few-shot-prompting-guide

    Example:
      Context → Making a cake: Several cake pops are shown on a display. A woman and girl
                are shown making the cake pops in a kitchen. They
    Correct Answer → bake them, then frost and decorate.
    Incorrect Answer → taste them as they place them on plates.
    Incorrect Answer → put the frosting on the cake as they pan it.
    Incorrect Answer → come out and begin decorating the cake as well.

    Context:
    'content':f"choose movie which fit description. \n"
              + f'<description> {user preferences} </description>\n'
              + movie_plot_context
    movie_plot_context -> <movie>1. ['Chain Reaction'
        'At the University of Chicago, a research team that includes brilliant student machinist Eddie Kasalivich experiences a breakthrough: a stable form of fusion that may lead to a waste-free energy source. However, a private company wants to exploit the technology, so Kasalivich and physicist Dr. Lily Sinclair are framed for murder, and the fusion device is stolen. On the run from the FBI, they must recover the technology and exonerate themselves.'
        "['Thriller', 'Action', 'Science Fiction']"]<movie>
    <movie>2. ['Grande, grosso e... Verdone' 'A comic movie divided in three episodes.'
        "['Comedy']"]<movie>
        ... <take more example randomly from dataset>
    Correct Answer -> ...
"""

for each in response.logprobs:
    token = each.get('token', '')
    logprob = each.get('logprob', None)
    alternatives_logprobs = each.get('top_logprobs', [])
    print(f'>> selected: {token}')
    print('alt: > ', end='')
    print([it['token'] for it in alternatives_logprobs])
    print('<<<<' * 40)

print(response.message.content)

"""
    3. Generation
"""
response_obj = Response.model_validate_json(response.message.content)
chosen_movie_idx = int(response_obj.best_fitting_movie)

chosen_movie_idx_global = closest_idxs[chosen_movie_idx - 1]
plot_overview = df_movies.iloc[chosen_movie_idx_global]['overview']
movie_title = df_movies.iloc[chosen_movie_idx_global]['original_title']

response = ollama.chat(
    model='gemma:2b',
    messages=[
        {
            'role': 'system',
            'content': f"you are the best movie critic ever existed."
        },
        {
            'role': 'user',
            'content': f"write short recommendation of movie based on preferences. \n"
                       + 'output 3-5 sentences, include movie title.\n'
                       + f'<description> {movie_desc} </description>\n'
                       + f'<movie_title> {movie_title} </movie_title>\n'
                       + f'<movie_overview> {plot_overview} </movie_overview>\n'
        }
    ]
    # ,format=Response.model_json_schema()
)

print(' Critic response: ')
print(response.message.content)