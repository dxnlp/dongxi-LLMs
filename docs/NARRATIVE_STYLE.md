# Narrative Style for Conceptual Chapters

Status: adopted 2026-10-20 at the learner's request, modeled on the narrative of
[Jurafsky & Martin, SLP3 Chapter 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf).
Piloted in [Chapter 10](../book/chapters/10-preferences-and-reward-models.md).
This guide changes how chapters *read*. It does not relax any evidence rule in
[BOOK_EDITORIAL_PLAN.md](BOOK_EDITORIAL_PLAN.md) or AGENTS.md.

## What the learner liked about SLP3

- An **epigraph** that frames the chapter's idea before any technical content.
- **Step-by-step entry**: a limitation of the previous stage motivates the new
  technique, and the chapter enters it one idea at a time.
- **Plain words before notation**, using everyday analogies and real examples.
- **Bold terms** on first definition, followed by a one-sentence plain meaning.
- **Algebra shown in steps**, not compressed into "it follows that".
- A closing **summary** of the main points.

## Rules

1. **Epigraph.** Open with a short, real, correctly attributed quotation as a
   blockquote. Name the work when the quote comes from a book. Use only quotations
   whose wording and attribution you are confident of, and don't invent the
   setting or date. Connect the quote to the chapter in the opening paragraphs.
2. **Open from the previous limitation.** The first paragraph says what the
   previous chapter could not do and why that matters, in concrete terms.
3. **Roadmap.** After the motivation, give a short numbered list of the steps the
   chapter takes, with section numbers.
4. **Section bridges.** Begin each section with two to four plain sentences that
   pose the problem the section answers, tied to what came before. Then give the
   precise treatment.
5. **Intuition before formula.** Give an analogy or a worked case in words first
   (chess ratings for Bradley–Terry, thermometers for the reward gauge, weather
   forecasts for calibration), then the definition.
6. **Define terms in bold** on first use, with a plain-language gloss.
7. **Show derivation steps.** When a line says "substitute" or "rearrange",
   display the intermediate line.
8. **Summary.** End with a "Chapter summary" of bullet points, each pointing back
   to its sections, placed before the companion route and exercises.
9. **Voice.** Short sentences, second person or "we", contractions are fine, and
   no stacked noun phrases ("native thinking-off cap 32 child") in teaching prose.
   Keep evidence detail, but lead into it with one plain sentence saying what the
   experiment shows.

## Preserve

Keep section numbers, heading text (for anchors and solution references), every
math expression, measured number, link, figure, code block and `chapter-gif`
block. Add prose; never weaken an evidence boundary to make a passage read
more smoothly. Run `scripts/check_book_math.py`, `scripts/check_book_prose.py`,
`scripts/check_course_integrity.py` and the book unit tests after each chapter.

## Rollout

| Chapter | Status |
|---|---|
| 10 Preferences and reward models | pilot, 2026-10-20 |
| 1–9, 11–15 | complete 2026-10-20; awaiting learner review |

All 15 chapters now have distinct epigraphs, opening roadmaps, section bridges and
a summary. Chapters 1 and 2 already had a numbered summary section after the
exercises, so the section-pointing bullets went inside it rather than adding a
second summary. A comparison against commit `1b00993` confirms that headings,
links, math blocks, inline math, Python code, measured numbers and `chapter-gif`
blocks are all preserved.

| Ch | Epigraph |
|---:|---|
| 1 | Feynman, "Cargo Cult Science" |
| 2 | J. R. Firth |
| 3 | Claude Shannon, "A Mathematical Theory of Communication" |
| 4 | William James, *The Principles of Psychology* |
| 5 | Feynman, Caltech blackboard |
| 6 | R. A. Fisher |
| 7 | William Bruce Cameron, *Informal Sociology* |
| 8 | Lewis Carroll, *Through the Looking-Glass* |
| 9 | Lewis Carroll, *Alice's Adventures in Wonderland* |
| 10 | Samuel Johnson, in Boswell's *Life of Johnson* |
| 11 | Thoreau, *Walden* |
| 12 | Aristotle, *Nicomachean Ethics* (Ross) |
| 13 | "Trust, but verify" (Russian proverb, via Reagan) |
| 14 | Strathern, paraphrasing Goodhart |
| 15 | Carl Sagan, *Cosmos* |

Epigraph wording was checked from memory, not against printed sources. Verify
each against a source before publication.
