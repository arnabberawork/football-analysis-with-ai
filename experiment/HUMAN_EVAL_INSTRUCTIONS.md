# Human evaluation — instructions for raters

Thank you for helping rate outputs from a football match-analysis system
for a Master's thesis. This should take about **15-25 minutes** for the
full sheet (20 pairs).

## What you're looking at

You'll rate pairs of AI-generated answers to football questions (match
summaries, "what did player X do", etc.), each grounded in real 2022 FIFA
World Cup match data. For each pair, you're shown **Response A** and
**Response B** — two different answers to the *same* question. You are
**not told which system produced which response** (this is a blind
comparison, by design) — just read both and rate each one independently
on its own merits.

You don't need to already know the outcome of the match to do this — the
query itself is shown, and you can judge coherence/fluency without
football knowledge. Factual consistency and relevance are easier to judge
well if you have some football familiarity or don't mind quickly checking
a score/scoreline online, but do your best either way.

## The file

`experiment/results/human_eval_sheet.csv` — open it in Excel, Google
Sheets, or any spreadsheet tool. Each row is one (query, response pair).
Columns:

| Column | What it is |
|---|---|
| `query` | The question both responses are answering |
| `response_a` / `response_b` | The two answers to rate |
| `relevance_a` / `relevance_b` | Your rating (fill these in) |
| `coherence_a` / `coherence_b` | Your rating (fill these in) |
| `factual_consistency_a` / `factual_consistency_b` | Your rating (fill these in) |
| `fluency_a` / `fluency_b` | Your rating (fill these in) |

You're filling in **8 numbers per row** (4 dimensions × 2 responses).
Leave everything else untouched.

## Rating scale: 1-5 for each dimension

Rate **Response A** and **Response B** separately on each of these four
dimensions. A tie (the same score for both) is a valid answer if you
genuinely can't tell them apart on that dimension.

**Relevance** — does the response actually answer what was asked?
- 1 = ignores the question / answers something else entirely
- 3 = partially answers it, or answers it but includes a lot of irrelevant padding
- 5 = directly and completely answers what was asked

**Coherence** — is it well-organised and easy to follow?
- 1 = disjointed, contradicts itself, hard to follow
- 3 = readable but a bit repetitive, disorganised, or awkwardly structured
- 5 = clear, logically ordered, easy to follow start to finish

**Factual consistency** — does it get the football facts right (scores,
scorers, minutes, cards, etc.), as far as you can tell?
- 1 = contains claims that are clearly wrong or made up
- 3 = mostly accurate but you're unsure about one or two specific details
- 5 = everything stated checks out / nothing looks invented

**Fluency** — is the writing itself natural, grammatical English?
- 1 = broken, garbled, or very awkward phrasing
- 3 = readable but stiff, repetitive wording, or minor grammar issues
- 5 = reads naturally, like a competent human wrote it

## What not to do

- Don't try to guess or research which underlying system produced A vs. B
  — it doesn't matter for your rating, and guessing wrong (or right) both
  bias the exercise.
- Don't leave any cell blank if you can help it — a genuine "I can't tell"
  is fine to express as a tied score (e.g. 3 and 3), not as blank.
- Don't rate the *question* itself, just the two responses to it.

## When you're done

Save the file and send it back. That's the only file you need — don't
worry about anything else in the `experiment/` folder.
