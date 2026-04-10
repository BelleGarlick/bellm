# Concept

> this doc is a stub that i'll com back to. it's largely just a colleciton of unorganised thoughts

Most llms use auto-regressive text prediction. Bellm's mean thing is diffusion.

In theory this means as the text if diffused, you have a conciousness of thought which anneals to a final thought. By predicting larger amounts of text we allow it to predict something resembling more of a thought than just text. Utilising the RL loop futhers this...

## Training Concepts

## Methodology

## RL Improvments

## Multi-Modality
Training multi-modality.
Inputs can be akin to prior inputs, outputs will be different and need a different loss function...
The output should deffuse as before on image/audio/video patches naturally in the diffusion output.

## Drawbacks
Large outputs
harder to train
Dont have a way to mask tokens as with auto-regressive. e.g. if we want to prevent a generation step from doing something, we could mask certain tokens in auto-regressive. here we cannot. The model will kinda of just think whatever it was.
    oh wait no we can do that by masking it throughout the diffusion process... hmm...