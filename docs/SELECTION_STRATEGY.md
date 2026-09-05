# Getting selected: what the judges said they want, and how EvoML answers

The buildathon page asks for a public repository, a 5-minute pitch video,
documented architecture, honest metrics, measured results, and an audit
trail. For the Open Track: a real problem, a working product, meaningful use
of AI.

## Map to the rubric

| Judges want | Where EvoML shows it |
|---|---|
| Honest metrics | random control on identical data, pre-registered gates, published caveats, Brier score, Chronos standalone score published even though it is a coin flip |
| Measured results | 14.6 days live, 572 resolved calls, z-scores, one SQL query reproduces everything |
| Audit trail | append-only SQLite, predictions logged before outcomes, evolution journal, live site fed by the running process |
| Working product | systemd service, dashboard, public site with live growth, 103 tests |
| Meaningful AI | self-evolving genome, from-scratch network, foundation-model ingestion measured against baselines |
| Fintech relevance | the loop generalises to fraud, risk and pricing models that decay; roadmap names a fraud-dataset bench |

## Ten things that raise the odds

1. **Keep the run alive until interviews.** A live scoreboard that has kept
   moving for 30+ days is more convincing than any slide. Do not reset the
   archive.
2. **Add the fraud bench (highest leverage).** Run the same genome
   tournament on a public fraud dataset (e.g. the IEEE-CIS or PaySim data)
   with a random control and time-ordered holdout. Even a modest, honest
   result proves the loop transfers to Razorpay's world.
3. **Publish the replication window.** Split the ledger into the first and
   second fortnight and report both; honesty about drift is the pitch.
4. **Record a 90-second walkthrough of the live site** (screen capture) and
   pin it in the README next to the cartoon pitch. Judges skim.
5. **Write a short technical post** (Dev.to / Medium / LinkedIn article):
   "Why noise dethroned my best model, and the one-standard-error rule that
   fixed it." Concrete lessons travel further than feature lists.
6. **Ship a `make verify` target** that runs the SQL reproduction and prints
   the gate table, so a judge can verify in one command.
7. **Add an issue tracker with 5 to 8 real open issues** (roadmap items).
   Signals a maintained project, not a hackathon dump.
8. **Tag a release** with the exact snapshot numbers and a frozen copy of the
   evolution journal (CSV) as a release asset.
9. **Be interview-ready on the failures**: the Claude arm that was stopped,
   the confidence inversion, the lost lineage at gen 371. Judges interviewing
   for an internship want to hear how you think when things break.
10. **One-line positioning for the panel**: "I built the harness that tells
    you whether your model still works, and a model that rewrites itself when
    it does not."

## Where to publish (in order of value)

1. GitHub: release v1.0.0, topics, description, pinned repo, social preview
   image (cover illustration).
2. YouTube (unlisted or public): the pitch video with chapters; paste the
   link into the submission and README. Judges can stream it without a
   download.
3. LinkedIn post tagging Razorpay and the buildathon hashtag, with the cover
   image and the three headline numbers.
4. X/Twitter thread: 6 posts, one per section of the pitch.
5. Show HN ("Show HN: EvoML, a self-evolving model with a random control and
   an audit trail") on a weekday morning US time.
6. Dev.to / Medium article (see item 5 above).
7. r/MachineLearning (Project flair) and r/algotrading, with the honesty
   framing front and centre; those communities reward caveats.
8. Hugging Face: a model card for EvoNet genomes is optional; the audience is
   smaller than the ones above.

Ready-to-paste copy for each channel is in `docs/PUBLISHING.md`.
