## V1 Capabilities
Basic diffusion proces. Input text, output text. Basic python coding. Basic Q+A style answering.
basic reasoning

## V2
More coding, image output? tool usage. Multi lingual

## Data Ingestion / Processing Pipeline
- V1
  - Have stuff for RL
  - Preprocess shards should be smaller
- V2
  - Foundation Model Datasets
    - reddit, wiki, leetcode, stackoverflow
    - coding ones and coding instruction ones, repositories etc
  - Unit Testing
  - Tool Usage
  - RL based format
  - Multi-Lingual
- V3
  - Image output
- V4
  - Video
  - Audio

## DataLoader
- V1
  - Docs
  - Loaders
    - Tokeniser loader
    - Foundation Loader
    - Instruction Loader
  - Parallel loading (should switch to multi-processing)
  - Auto-tokenisation
  - Unit testing
- V2
  - Utilise S3

## Tokeniser
- V1
  - Docs
    - Training
    - Pruning Samples
    - Pruning Sizes
  - redo with resoning, assistant, user tokens
  - Working with the datasets/dataloaders
- V2 Onwards
  - Retrain the tokeniser per run

## Training
- V1 
  - Tensorboard metrics
  - Model definition defined better
  - Interact with model as it's training via checkpoints
  - Save to some assets store with tags and expirary
  - Flex attention? Longformer? - 5k context size
  - Masking
  - 12 transformer layers
- V2
  - Sage maker compatible
  - 25k context size
  - 24 transformer layers
- V3
  - Distributed training
  - 36 transformer layers
  - 125k context size
  - images input/output
  - different loss for different modalities of outputs with masking?
  - pretrained var
- V4
  - text, images, video, text, audio
  - diffusing on these might be difficult, need some parameter formarking the type of veector that the embedding is, both input and output. might need boundary token for this.
    - possibly apply different losses for each slice of the tensor

## Instruction Fine Tuning
- V1
  - Basic Q/A chat conversations with reasoning
  - unit tests
- V2
  - Incliude q+a style posts like on reddit or otherwise. more datasets and better reasoning?
  - sage maker compatible
  - basic tool usage
- V3
  - Distributed
  - Tool usage

## RL Environment
- V1
  - Knowing if it knows something or not
  - Basic facts questions, need some kinda facts database. sometimes environment automatically outputs curent response soit knows whats corect
  - Basic rl environment
  - basicl rl reward mechanism
  - basic python easy leet code questions
  - normally run using model samlping
    - occasionally output optimal items if available
  - basic value function/training stuff
    - every n episodes recompute the value function
  - with text also output the code
  - need q+a style stuff
- v2
  - coding
    - langs: pthon, javascript/typescript, java, c++
    - leetcode easy - hard
    - have rl evaluate how good it is or correct it is and reward
  - More general knowledge q+a, also a way to know whether it knows or not
  - sage maker compatible
- v3
  - should have ay to create the question randomly
  - maths
  - physics
  - chemistry
  - biology
  - language translations
  - expand to more diciplines
  - distirbuted
  
Future work should also then distill / train smaller models