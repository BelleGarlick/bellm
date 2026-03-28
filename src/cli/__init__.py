from bellm.tokeniser import Tokeniser

text = """
I think to me a lot of my framing around this is having been around the BDSM community a while, pet play is inherently linked sexual aspects. I guess, I have never met a puppy girl who doesn't also involve it in sex, and I am trying to frame the separation here. I'm not trying to be obtuse I am genuinely asking and showing where I am coming from. I am not trying to see that all expressions related to being a puppy is kink. 
We are a moderated space, and I just think there is some kind of middle ground really.  Maybe a better example is, if I am a nudist by identity, is it still fair to be naked in front of my friends if it is me expressing my identity. I don't think there is a clear cut answer, and I am not saying anyone has to change, but seeing the constant chat about it is alienating to people too
i had a lengthy message written but after reading through it felt overly confrontational and unhelpful and i did not feel comfortable sending it as it was so i am going to try again from a bit of a chiller point of view
i do not know how much i agree with calling one of them scientifically proven and one not. not to say i dont believe science, but i know what i am and i do not believe i need a secondary source to validate what i feel as i am. i think scientific bodies at large do not have my trust, especially when we take into account things like how often the trans community is waylaid in spite of provable science. especially when marginalised communities already suffer at rhe hands of scientists.
i respect your view on being a woman being different from transhumanism. i do not know how much i agree genuinely, but i think there is a degree of comparison that can still be made and have validity.
i dont think pet play is inherently linked. i am entrenched in a lot of those cultures, i know folk who do engage in it but i also know equal amounts not. i think it would be a fair assessment to make that a bdsm communities ties to something are going to be flavoured a lot more that would engage it on that lens, but i know tonnes of folks that engage with therian shit on a level that is not kink. 
i do not know what the, if there is, a fix for it being traumatic for you. i am sorry that it is, especially seeing as it is such a common thing in the world at large. i do want to ask how much of rhe problem is kink based, though. dogs are not the only brand of furries / therians, there are a significant number of folks on this server who identify in differing ways, that do not seem to have this issue. is it that the more esoteric it is the less easy it is to see its kink? for some dollhood is a kink, must i censor a fair part of who i am because rhats how i am seen to others? i mean, one of the things i got asked when i joined here is if i meant a sex doll (presumably as a joke). there are also some that identify as things as a joke, are they allowed to do so because it’s not taken seriously? 
i am genuinely really sorry that you have this as something that is traumatic. i imagine it must be a hard tiing to deal with, both in and outside of communities like this. i just am not sure the fix is to call all puppyhood kink in an attempt to soft blanket ban it. i know you do not agree but to some, to me, this is a near integral part of the identity. to pretend i am not is to accept me for less than who i am, and i am certain there are others that feel this way
i really hope this does not come across as particularly rude or offensive or anything like that, if so i apologise i genuinely mean this not in an argumentative way but as a constructive conversation to be had here
"""

# print(text[420:440])


tokeniser = Tokeniser().load(f"../bellm/tokeniser/tokeniser.json")


colors = [
    "\033[38;2;0;0;0;48;2;112;230;255m",
    "\033[38;2;0;0;0;48;2;255;230;112m",
    "\033[38;2;0;0;0;48;2;255;112;166m",
    "\033[38;2;0;0;0;48;2;112;255;151m",
    "\033[38;2;0;0;0;48;2;255;151;112m",
    "\033[38;2;0;0;0;48;2;233;255;112m",
]


if __name__ == "__main__":
    for item in list(tokeniser.token_map):
        if tokeniser[item] > 15_000:
            del tokeniser.token_map[item]

    tokens = tokeniser.tokenize(text)
    for i, token in enumerate(tokens.tokens):
        print(colors[i % len(colors)] + str(token), end="")
