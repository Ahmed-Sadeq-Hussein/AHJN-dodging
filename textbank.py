r"""
textbank.py -- the sentences used for training and evaluation, kept apart.

Design A and B trained on five sentences and were evaluated on the same five.
Any policy learned under that setup could memorise per-sentence structure and
still look like it generalised, because the held-out axis was voices only.
Distillation makes that risk much worse: it fits a supervised target
directly, so it will happily learn "this exact utterance needs this exact
edit" if the text never varies.

So the bank is split. TRAIN_TEXTS is what PGD targets are collected on;
EVAL_TEXTS is never seen during collection or distillation and only appears
at final evaluation, alongside held-out voices. A result is only interesting
when it holds on unseen voice AND unseen text.
"""

from __future__ import annotations

TRAIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the morning light spills across the valley.",
    "She sold seashells by the seashore every summer for as long as anyone could remember.",
    "There is nothing more deceptive than an obvious fact when you have not looked closely at it.",
    "The train arrives at platform nine just before midnight, and the station is almost empty.",
    "Rain fell steadily on the tin roof, and the whole house smelled of woodsmoke and coffee.",
    "He counted the coins twice, then a third time, before pushing them across the wooden counter.",
    "Nobody had opened that door in eleven years, and the hinges had rusted almost solid.",
    "The map was wrong about the river, but right about almost everything else that mattered.",
    "Every winter the lake freezes from the eastern shore first, and the children know it.",
    "She wrote the letter four times and burned three of them in the kitchen sink.",
    "The engine coughed twice, caught, and settled into a rhythm that sounded almost healthy.",
    "Old books smell of vanilla because the paper breaks down slowly into sweeter compounds.",
    "They agreed to meet at the bridge, though neither of them expected the other to come.",
    "The measurement was off by a fraction of a degree, which by the coast meant miles.",
    "Light from that star left before anyone alive had thought to give it a name.",
    "He learned the song by ear, badly, and played it anyway for the rest of his life.",
    "The garden had gone wild in three seasons, and honestly it looked better for it.",
    "Salt, patience, and a very sharp knife are most of what the recipe actually requires.",
    "She could tell from the footsteps on the stair exactly who was about to knock.",
    "The contract ran to ninety pages, and the important sentence was buried on page sixty.",
    "Snow fell all night without any wind, so by morning the fences had disappeared entirely.",
    "There were three keys on the ring and none of them opened anything he owned.",
]

EVAL_TEXTS = [
    "The clock in the hallway had been running four minutes fast since the war ended.",
    "Bring the lantern closer, because I cannot read this handwriting in the dark at all.",
    "Whatever happens tomorrow, remember that the tide comes in twice and asks nobody first.",
    "A good argument changes your mind, and a great one changes what you look at.",
    "The dog waited by the gate every evening for a man who was not coming back.",
    "Steam rose off the road as the sun came over the ridge and dried the rain.",
]

ALL_TEXTS = TRAIN_TEXTS + EVAL_TEXTS

# Voices split the same way. The held-out four are chosen to overlap the
# randomly-drawn evaluation set from the A-vs-B comparison, so the two
# experiments' held-out numbers are directly comparable.
TRAIN_VOICES = ["Ego", "Joffrey", "Silco", "Tywin", "ahmed", "david",
                "jhin", "riven", "samira", "swain", "tahmkench"]
EVAL_VOICES = ["Tywin_2", "evelynn", "mordekaiser", "thresh"]
