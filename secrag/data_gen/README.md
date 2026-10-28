# `secrag.data_gen` — synthetic corpus and QA set

Generates the knowledge base of the fictional **Northbridge Institute of Data Practice**
and the questions we ask about it. Fixed seed → byte-identical output (decision D14).

```
python -m secrag generate-data --seed 13
```

| File | What it holds |
|---|---|
| `world.py` | Entities (staff, campuses, courses, membership tiers, events, policy limits) drawn from a seeded RNG; `build_facts` turns them into `Fact`s |
| `facts.py` | `Fact` record and surface forms (`AUD 1,285` / `$1,285.00`, `15 March 2027` / `15/03/2027`, ...) |
| `builder.py` | `CorpusBuilder.say(draft, fact_id)` renders a fact **and** records that the draft states it, so gold labels are exact |
| `documents.py` | 76 core pages: course pages, fee schedule, tier pages, policies, directories, profiles, campuses, events, FAQs, news |
| `distractors.py` | 150 distractors: knowledge-hub articles, member-forum posts (6 quote stale values), committee minutes, job ads |
| `qa.py` | 100 answerable questions (one fact each, phrasing varied) + 15 unanswerable ones |

Outputs:

- `data/corpus/northbridge/*.md`: what the pipeline reads (front-matter + markdown).
- `data/qa/northbridge_qa.jsonl`: questions with gold documents and answers (**evaluator-only**, D12).
- `data/qa/northbridge_facts.jsonl`: every fact, where it is stated, and any stale value (**evaluator-only**).

Design notes:

- Values are unique within a kind where it matters (no two fees or event dates coincide), so a matched value
  identifies its fact.
- Emails use `.example` domains; phone numbers use the 5550 xxxx range reserved for fiction.
- Trust levels follow the source: policy library / course catalogue = official, staff pages = staff,
  member forum and job board = unverified.
- `tests/test_data_gen.py` checks that every gold document really contains its answer and that the committed
  `data/` folder equals the generator's output.
