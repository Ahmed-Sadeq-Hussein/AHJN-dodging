# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
prosody.py -- the style variants tested in the "does dramatising help?" experiment.

Five renditions of identical content, so any difference in detector score is
attributable to delivery rather than to what was said:

  neutral      plain text, the regime every earlier result used
  pauses       punctuation-driven timing: commas, ellipses, em dashes
  emphasis     <strong> tags on the stressed words
  breathy      <strong> plus [breath] / [quick_breath] between clauses
  instruct     natural-language style prompting via inference_instruct2

The first four change the TEXT that reaches the tokenizer; the fifth changes
the CONDITIONING and leaves the text alone. They are different mechanisms and
are reported separately, because "expressive markup" succeeding while
"expressive instruction" fails (or the reverse) says something specific about
where the detectable artefacts come from.

Why this might move detector scores at all: the four anti-spoofing models are
trained overwhelmingly on read speech. Expressive prosody is out of
distribution for both the bonafide and spoofed halves of their training data,
so it could read as more human -- or expressive synthesis could push the
vocoder into wider F0 excursions and more artefacts, and read as less. The
direction is genuinely not predictable from first principles, which is why it
is worth measuring rather than arguing about.
"""

from __future__ import annotations

# Content is held constant across styles. Each entry is
#   (neutral, pauses, emphasis, breathy)
# and the instruct variant reuses the neutral text with an instruction.
VARIANTS: list[dict[str, str]] = [
    {
        "neutral": "I sentence you to death, for your transgressions.",
        "pauses": "I sentence you... to death... for your transgressions.",
        "emphasis": "I <strong>sentence</strong> you to <strong>death</strong>, "
                    "for your <strong>transgressions</strong>.",
        "breathy": "I <strong>sentence</strong> you [breath] to <strong>death</strong>, "
                   "[quick_breath] for your <strong>transgressions</strong>.",
        "breath_only": "I sentence you [breath] to death, [quick_breath] "
                       "for your transgressions.",
        "neutral_long": "I sentence you to death, for your transgressions. "
                        "The court has heard the evidence and reached its decision.",
    },
    {
        "neutral": "You will not speak to me that way again, not in this house.",
        "pauses": "You will not speak to me that way again... not in this house.",
        "emphasis": "You will <strong>not</strong> speak to me that way "
                    "<strong>again</strong>, not in <strong>this</strong> house.",
        "breathy": "You will <strong>not</strong> speak to me that way "
                   "<strong>again</strong>, [breath] not in <strong>this</strong> house.",
        "breath_only": "You will not speak to me that way again, [breath] "
                       "not in this house.",
        "neutral_long": "You will not speak to me that way again, not in this house. "
                        "I have tolerated a great deal, but that was the last of it.",
    },
    {
        "neutral": "Everything I built, everything I protected, and this is how it ends.",
        "pauses": "Everything I built... everything I protected... and this is how it ends.",
        "emphasis": "<strong>Everything</strong> I built, <strong>everything</strong> "
                    "I protected, and <strong>this</strong> is how it ends.",
        "breathy": "<strong>Everything</strong> I built, [quick_breath] "
                   "<strong>everything</strong> I protected, [breath] and "
                   "<strong>this</strong> is how it ends.",
        "breath_only": "Everything I built, [quick_breath] everything I protected, "
                       "[breath] and this is how it ends.",
        "neutral_long": "Everything I built, everything I protected, and this is how "
                        "it ends. Not with a battle, but with a signature on a page.",
    },
    {
        "neutral": "Tell me again how you had no choice in the matter.",
        "pauses": "Tell me again... how you had no choice... in the matter.",
        "emphasis": "Tell me <strong>again</strong> how you had <strong>no choice</strong> "
                    "in the matter.",
        "breathy": "Tell me <strong>again</strong> [breath] how you had "
                   "<strong>no choice</strong> in the matter.",
        "breath_only": "Tell me again [breath] how you had no choice in the matter.",
        "neutral_long": "Tell me again how you had no choice in the matter. "
                        "I would very much like to hear that explanation twice.",
    },
    {
        "neutral": "The gates are open, the guards are gone, and nobody is coming to help.",
        "pauses": "The gates are open... the guards are gone... and nobody is coming to help.",
        "emphasis": "The gates are <strong>open</strong>, the guards are "
                    "<strong>gone</strong>, and <strong>nobody</strong> is coming to help.",
        "breathy": "The gates are <strong>open</strong>, [quick_breath] the guards are "
                   "<strong>gone</strong>, [breath] and <strong>nobody</strong> "
                   "is coming to help.",
        "breath_only": "The gates are open, [quick_breath] the guards are gone, "
                       "[breath] and nobody is coming to help.",
        "neutral_long": "The gates are open, the guards are gone, and nobody is coming "
                        "to help. We are entirely on our own until the morning.",
    },
    {
        "neutral": "Do you understand what you have cost me, after all these years.",
        "pauses": "Do you understand... what you have cost me... after all these years.",
        "emphasis": "Do you <strong>understand</strong> what you have <strong>cost</strong> "
                    "me, after <strong>all</strong> these years.",
        "breathy": "Do you <strong>understand</strong> [breath] what you have "
                   "<strong>cost</strong> me, [quick_breath] after <strong>all</strong> "
                   "these years.",
        "breath_only": "Do you understand [breath] what you have cost me, "
                       "[quick_breath] after all these years.",
        "neutral_long": "Do you understand what you have cost me, after all these years. "
                        "I do not think you have ever once considered it properly.",
    },
]

# Style prompts for inference_instruct2. The calm one is a control: if BOTH
# instruct variants move the score equally, the effect is the instruct
# pathway itself (which replaces the reference transcript with the
# instruction, changing the conditioning) rather than the emotion in it.
INSTRUCTIONS = {
    "instruct_dramatic": "Speak with rising anger and heavy dramatic emphasis, "
                         "raising the voice towards the end of each phrase.",
    "instruct_calm": "Speak calmly and evenly, in a flat measured tone.",
}

# Two controls for the duration confound. The breathy renditions run roughly
# twice as long as neutral, and the detectors tile-repeat anything under
# their 64600-sample (4.04 s) window -- so a bare neutral/breathy comparison
# cannot tell "breath sounds read as human" apart from "the clip finally
# filled a window without padding".
#
#   neutral_long  same register, no tags, extended to breathy's duration
#   breath_only   [breath] tags with no <strong>, isolating breath from stress
TEXT_STYLES = ("neutral", "pauses", "emphasis", "breathy",
               "neutral_long", "breath_only")
ALL_STYLES = TEXT_STYLES + tuple(INSTRUCTIONS)


def render(variant: dict[str, str], style: str) -> tuple[str, str | None]:
    """-> (text to synthesise, instruction or None)."""
    if style in INSTRUCTIONS:
        return variant["neutral"], INSTRUCTIONS[style]
    return variant[style], None


def breathify(text: str, every: int = 1) -> str:
    """Insert breath tokens at clause boundaries.

    The hand-written VARIANTS above are for measurement; a training corpus
    needs the same treatment applied to arbitrary sentences. Breaths go after
    commas and mid-sentence full stops -- where a speaker actually takes one
    -- alternating [breath] and [quick_breath] so the corpus is not a single
    token repeated. `every` skips boundaries to thin the density.

    Never appends a breath after the final word: a trailing inhale at the end
    of an utterance is the one placement that reliably sounds wrong, and it
    would lengthen every clip in a way that reintroduces exactly the duration
    confound the neutral_long control was added to rule out.
    """
    import re

    parts = re.split(r"([,;.])", text)
    out, hit = [], 0
    for i, part in enumerate(parts):
        out.append(part)
        if part in ",;." and i < len(parts) - 2:
            if "".join(parts[i + 1:]).strip():
                hit += 1
                if hit % every == 0:
                    out.append(" [breath]" if hit % 2 else " [quick_breath]")
    result = " ".join("".join(out).split())

    # Eight of the 22 training sentences carry no internal punctuation, so a
    # purely punctuation-driven rule would leave a third of the corpus
    # untreated and silently weaken the contrast the experiment is testing.
    # Fall back to a single breath at the nearest word boundary to the
    # midpoint, which is where an unpunctuated clause tends to break anyway.
    if hit == 0:
        words = result.split()
        if len(words) >= 6:
            mid = len(words) // 2
            words.insert(mid, "[breath]")
            result = " ".join(words)
    return result
