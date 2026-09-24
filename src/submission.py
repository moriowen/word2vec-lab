"""The submission layout: the course's AI-assisted HW template wrapped around the report.

The template asks, for each question, for Step 1 (the final answer) and Step 2 (the AI
interaction record: tool, prompt, round-by-round workflow, strategy, critique), then Step 3
(references) and Step 4 (appendices) once at the end. Step 1 reuses the report sections that
build_report.py already renders, so the web page and the PDF never disagree on a number.
Only build("submission") imports this module; the web page is untouched by it.
"""
import json
from types import SimpleNamespace

# Each handout deliverable is one "question". `step1` names report sections, in order.
QUESTIONS = [
    ("q1", "Question 1", "Deliverable 1",
     "Compare two Word2Vec models, describing each one's neural architecture from input to "
     "output with a figure", ["results", "arch", "scratch"]),
    ("q2", "Question 2", "Deliverable 2, parts a to c",
     "Describe the training and test data; give the pretrained model's reference, URL and "
     "accuracy; five training and four test examples for the fine-tune; a performance table "
     "with correct and incorrect examples", ["data", "finetune", "errors"]),
    ("q3", "Question 3", "Deliverable 3",
     "Top-K ranked results for five test examples, compared against a pretrained model",
     ["testcases"]),
    ("q4", "Question 4", "Deliverable 4",
     "Performance measurements for training and for testing, per query, and a comparison of "
     "the models", ["systems"]),
    ("q5", "Question 5", "Deliverable 5",
     "Analyze the models' hyperparameters and compare top-1, top-5 and top-10 ranked words "
     "for five word queries", ["hyper", "neighbours"]),
    ("q6", "Question 6", "Deliverable 6",
     "Experience with installation, running and measurement", ["running"]),
    ("q7", "Question 7", "Handout steps: query, test and visualize",
     "Find similar words with most_similar, try vector arithmetic analogies, and project the "
     "vector space to two dimensions with PCA or t-SNE", ["analogies", "viz"]),
]
APPENDIX = [("appA", "Appendix A", ["similarity"]), ("appB", "Appendix B", ["text8"])]
FRONT = [("preface", "How this document is organised"), ("tools", "Shared AI interaction record"),
         ("pace", "Compute environment: PACE ICE")]

SITE = "https://word2vec-lab.vercel.app/"
REPO = "https://github.com/moriowen/word2vec-lab"
# The PDF build swaps this for the STUDENT_ID environment variable; the ID is never stored.
SID_SLOT = '<span class="sid" data-slot="student-id"></span>'


def placed():
    return ([i for *_, ids in QUESTIONS for i in ids] + [i for *_, ids in APPENDIX for i in ids])


def toc(titles):
    out = ['<div class="tg">Front matter</div>']
    out += [f'<a href="#{i}">{t}</a>' for i, t in FRONT]
    for qid, label, _, _, _ in QUESTIONS:
        out.append(f'<a href="#{qid}" class="tq">{label}</a>')
    out.append('<div class="tg">End matter</div><a href="#refs">Step 3: References</a>')
    out += [f'<a href="#{aid}">{label}: {titles[ids[0]]}</a>' for aid, label, ids in APPENDIX]
    return "".join(out)


def cover():
    return f'''<div class="tcover">
      <div><span>Course title</span>CS 6220 Big Data Systems and Analytics, Fall 2026, session cs6220-A</div>
      <div><span>HW2</span>Programming [X] &nbsp; Reading Critique [ &nbsp;]</div>
      <div><span>Student name</span>Atharva Mohite</div>
      <div><span>Student ID</span>{SID_SLOT}</div>
      <div><span>Live site</span><a href="{SITE}">{SITE}</a></div>
      <div><span>Repository</span><a href="{REPO}">{REPO}</a></div>
    </div>'''


def contents(titles):
    rows = [(f'<a href="#{i}">{t}</a>', "") for i, t in FRONT]
    rows += [(f'<a href="#{qid}">{label}: {dl}</a>',
              ", ".join(titles[i] for i in ids)) for qid, label, dl, _, ids in QUESTIONS]
    rows.append(('<a href="#refs">Step 3: References</a>', "Outside the page limit"))
    rows += [(f'<a href="#{aid}">{label}: {titles[ids[0]]}</a>', "Outside the page limit")
             for aid, label, ids in APPENDIX]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
    return (f'<section id="contents"><div class="tablewrap"><table class="toc-table"><thead><tr>'
            f'<th>Section</th><th>Step 1 draws on</th></tr></thead><tbody>{body}</tbody>'
            f'</table></div></section>')


# ---------------------------------------------------------------- front matter

def front(c):
    t8 = c.r["C-text8"]
    return f'''<section id="preface">
      <div class="sechead"><div class="eyebrow">Front matter</div>
      <h2>How this document is organised</h2></div>
      <p class="prose">The course template asks for four steps for every question. The
      handout's Deliverable section lists six items, and the handout also names three steps
      outside that list: find similar words, try vector arithmetic, and plot the vector space.
      I treat each of the six deliverables as one question and the three extra steps as a
      seventh. Each question gives Step 1, the final answer, followed by Step 2, the AI
      interaction record. Step 3 (references) and Step 4 (appendices) come once, at the end.</p>
      <p class="prose">Step 2 items that are the same for every question, which are the tool
      and the overall way I used it, are stated once below. Each question then gives its own
      summary of what I asked, the rounds that produced the answer, my prompting strategy and
      my critique of what the AI got right and wrong.</p>
      <p class="prose">Every number in Step 1 is read from <code>results/*.json</code> when the
      document is built, never typed into the text. The same sections, without the coursework
      framing, make up the live site at <a href="{SITE}">{SITE}</a>.</p></section>

    <section id="tools">
      <div class="sechead"><div class="eyebrow">Step 2, item 1, for all seven questions</div>
      <h2>Shared AI interaction record</h2></div>
      <p class="prose">Two kinds of model appear in this project, and they have to be kept
      apart. The <b>assistant</b> is the AI tool that helped me plan the experiment, write the
      code and draft the text. The <b>subjects</b> are the Word2Vec models being measured.
      They are not generative AI, and every number in the tables comes from them, through code,
      never from the assistant.</p>
      {c.table(["Tool", "Role", "What I used it for", "URL"], [
        ["Claude Code (Anthropic), running Claude Opus 5 and Opus 5.5, with Sonnet 5 for "
         "repository chores", "Assistant",
         "Reading the handout and planning, the Python pipeline, the from-scratch PyTorch "
         "trainer, the PACE ICE job script, the analysis, the report builder and this "
         "document. Every Step 2 record below comes from these sessions.",
         "<code>https://claude.ai/code</code>; API <code>https://api.anthropic.com/v1/messages</code>"],
        ["Word2Vec models A, B, C, G and G-ft", "Subject",
         "The objects of study. A, B and G-ft were trained with gensim, C with my own PyTorch "
         "code, and G was downloaded pretrained.",
         "gensim: <code>https://radimrehurek.com/gensim/</code>; GoogleNews vectors: "
         "<code>https://code.google.com/archive/p/word2vec/</code>"],
      ], ["l", "m", "l", "l"])}
      <p class="prose">One rule held throughout: no output from the assistant becomes data.
      Gold labels come from SST-2, similarity ratings from the published benchmarks, and every
      score from code that runs the subject models. When the assistant wrote a sentence
      claiming a number, the number was read from the result files at build time, so a claim
      that stopped matching the data would show as a mismatch rather than survive silently.
      Several of the critiques below are about the cases where the prose went wrong anyway.</p>
      <p class="prose">The work ran across several Claude Code sessions between Sep 22 and
      Sep 24, 2026. The dated run log in <code>IMPLEMENTATION.md</code> records each phase, and the
      commit history records every change.</p></section>

    <section id="pace">
      <div class="sechead"><div class="eyebrow">Front matter</div>
      <h2>Compute environment: Georgia Tech PACE ICE</h2></div>
      <h3 class="subhead">The short version</h3>
      <p class="prose">Everything ran on my M2 MacBook Air (8 GB) except one job. Model C on
      text8 has a {t8["train"]["vocab_size"]:,}-word vocabulary and
      {t8["train"]["corpus_tokens"]:,} tokens, and on the laptop it was on course for about
      four hours while leaving the machine unusable. I moved that one job to a PACE ICE A100
      as a Slurm batch job. It finished in {t8["train"]["wall_s"]:.0f} seconds. Unlike HW1,
      which tunnelled an interactive Ollama server, nothing here needs a live connection: the
      job runs on its own and writes a result file I copy back.</p>
      <h3 class="subhead">The longer version</h3>
      <ol class="steps">
        <li><div class="step-b"><b>Connect the GT VPN and install an SSH key</b>
          <span class="why">A key on the login node let the assistant run <code>rsync</code>,
          <code>sbatch</code> and <code>squeue</code> without a password prompt in every
          command.</span></div></li>
        <li><div class="step-b"><b>Copy the code to scratch and build a virtual environment</b>
          <span class="why"><code>module load python/3.11.9</code>, then a separate
          <code>.venv-ice</code> with the CUDA 12.6 build of PyTorch. The first attempt used
          the CUDA 13.0 build; swapping it in place with <code>--no-deps</code> broke the
          install on a missing <code>libcufile.so.0</code>, so the environment was rebuilt
          from scratch.</span></div></li>
        <li><div class="step-b"><b>Submit <code>slurm/c_text8.sbatch</code></b>
          <span class="why">It asks the <code>ice-gpu</code> partition for one GPU of a named
          type, 4 cores, 32 GB and 3 hours. The type has to be named: the partition also holds
          AMD MI210 cards, which the CUDA build of PyTorch cannot use.</span></div></li>
        <li><div class="step-b"><b>Check the GPU before training</b>
          <span class="why">The script prints <code>nvidia-smi</code> and runs a small CUDA
          computation first. On two L40S nodes that check failed with
          <code>uncorrectable ECC error</code>, so the job stopped in seconds instead of
          training on faulty hardware. The run that succeeded was resubmitted on an A100.</span></div></li>
        <li><div class="step-b"><b>Copy the result back</b>
          <span class="why">The job writes <code>results/C_text8.json</code>, labelled with the
          machine it ran on (<code>{e(t8["machine"])}</code>), and the report reads it like any
          other record.</span></div></li>
      </ol>
      <p class="prose">Two design decisions made the cluster run trustworthy. The trainer saves
      a checkpoint after every epoch (model, optimizer, random state and progress), written
      to a temporary file and renamed, so a job killed by the time limit resumes from its last
      finished epoch. I checked that a resumed run gives the same weights as an uninterrupted
      one. And the hyperparameters did not change with the hardware: the laptop run, stopped
      after two epochs, had reached a loss of {t8["train"]["loss_curve"][1]} at epoch 2, and so
      did the A100 run. The GPU changed the speed, not the result.</p></section>'''


def e(x):
    import html
    return html.escape(str(x))


# ---------------------------------------------------------------- step 2, per question

def records(c):
    acc, sw, r = c.acc, c.sw, c.r
    pt = c.corpus["phrase_train"]
    tcat = c.corpus["test_categories"]
    test_share = (tcat["negation"] + tcat["contrast"]) / sum(tcat.values())
    share = lambda m: ((c.err(m)["error_categories"].get("negation", 0)
                        + c.err(m)["error_categories"].get("contrast", 0)) / c.err(m)["n_errors"])
    shares = [share(m) for m in c.order]
    sgns, cbhs = sw["skip-gram + NS"]["acc"]["mean"], sw["CBOW + HS"]["acc"]["mean"]
    viz = c.viz
    ev = [sum(viz[m]["pca"]["explained_variance"]) for m in viz]
    return {
    "q1": dict(
        asked="Read the handout, propose a set of models that goes beyond the minimum without "
              "being impossible in three days, then build and compare them.",
        rounds=[
            ("The handout PDF", "Analyse the assignment and plan something larger than "
             "required that fits three days", "A first plan in <code>PLAN.md</code>."),
            ("That plan", "I am still unclear what is expected. Give a simple summary, say "
             "what goes beyond the requirement, and note that PACE ICE is available",
             "The plan rewritten around five models: A (skip-gram), B (CBOW), G (GoogleNews), "
             "G-ft (fine-tuned) and C (from scratch)."),
            ("The approved plan", "Write a step-by-step implementation plan; start only on my "
             "approval", "<code>IMPLEMENTATION.md</code>, 13 phases, each with an acceptance "
             "check."),
            ("Phase 10", "Implement skip-gram with negative sampling from scratch",
             "The first run did not learn. The assistant traced it to the loss reduction and "
             "fixed it."),
            ("An independent review of the repository that I ran separately",
             "Address its findings", f"A2 added as a control. The warm start is worth "
             f"{acc('G-ft') - acc('A2'):+.3f}, not {acc('G-ft') - acc('A'):+.3f}."),
            ("The finished report", "Why does the model with more data do worse? Keep the "
             "Google models last and pair A (SST) with A (text8)",
             "The cards and tables regrouped by corpus, with each model's training data "
             "stated on its card."),
        ],
        strategy=f'''<p>I split the work into a plan I could argue with and a build I could
          check. The first plan was produced from the handout alone, and I did not accept it:
          I asked for a plain summary of what was required, what was extra and how each part
          would be done. Only after that did I ask for an implementation plan, and I asked for
          it to wait for my approval. Every phase in that plan ends with an acceptance check
          stated before the code exists, such as "<code>most_similar('good')</code> returns
          sensible neighbours" or "accuracy lands in a known range". That turned "does it
          work?" into a question with a yes or no answer.</p>
          <p>I approved a model set in which pairs differ in as few things as possible.
          A and C share everything but the implementation. A and G-ft were meant to share
          everything but the starting weights, which is why the missing control mattered so
          much once it was pointed out. I kept the from-scratch model in scope and asked for
          it early, because the handout asks for an explanation of the architecture and
          writing the training loop is the most direct way to earn one.</p>''',
        critique=f'''<p><b>What it got right.</b> The architecture explanation is correct and
          careful. It describes Word2Vec as two matrices and a dot product, with no hidden
          layer, and it points out that the handout's encoder, code and decoder vocabulary
          fits Word2Vec only loosely, since the model predicts a neighbour rather than
          reconstructing its input. The from-scratch trainer includes the five pieces that
          matter (subsampling, the unigram^0.75 noise table, a dynamic window, two matrices
          and learning-rate decay), and the assistant checked two of them numerically before
          training anything.</p>
          <p><b>What it missed.</b> Three things. First, A against B changes two factors at
          once, architecture and loss, so the pair on its own cannot say which one matters.
          The assistant presented the gap as an architecture effect. The sweep I asked for
          later settled it: architecture matters and the loss does not. Second, it presented A
          against G-ft as a clean test of the warm start when G-ft also trains with a lower
          learning rate and fewer epochs. The independent review caught this, and the
          controlled comparison made the effect four times larger than claimed. Third, the
          headline gaps are single runs. A and B differ by {acc("A") - acc("B"):.3f} in the
          headline table but by {sgns - cbhs:.3f} across three seeds, so the first number
          overstated the difference by about half.</p>
          <p><b>What went wrong and was fixed.</b> The first from-scratch run sat at a loss of
          exactly 1.3863 for ten epochs. The plan had warned about a different bug (a wrong
          noise table, where the loss falls but the vectors are useless), so the prediction was
          wrong, but the check that caught the real bug, gating on accuracy instead of the
          loss curve, was the one the plan had set up.</p>'''),
    "q2": dict(
        asked="Choose and document the data, then cover parts a to c of deliverable 2 "
              "and check the report against the handout's wording.",
        rounds=[
            ("Phase 1 of the plan", "Continue with the next phases",
             "The assistant found that the GLUE copy hides every test label as -1, switched to "
             "SetFit/sst2, and added a check that fails if the labels are hidden."),
            ("First results: 134k tokens, noisy neighbours", "What's next? (the assistant "
             "proposed the change)", "Training moved to the phrase-level release, 67,348 rows. Accuracy rose by 8 "
             "points for A and 10 for B."),
            ("The tracked files and the handout", "Is any task still remaining?", "Part 2b was missing entirely; five training and four test "
             "examples were added, chosen by a stated rule."),
            ("The independent review", "Address its findings",
             "A reference table for the pretrained model (2a) and an explicit correct and "
             "incorrect example set for G-ft (2c)."),
            ("The live site", "Why does the website have no links or mention of the data? "
             "Add a section on what the data looks like, with metrics",
             "A data section built from <code>results/corpus.json</code>."),
            ("The data section", "How was the training so small? Something seems off",
             f"The assistant measured that {pt['rows_inside_frac']:.1%} of phrase rows are "
             f"substrings of the sentence release, and the report now says so."),
        ],
        strategy=f'''<p>For the data I asked questions rather than giving instructions. "What
          was the dataset for movie reviews?", "why does the website have no mention of the
          data?" and "how was the training so small?" each exposed something the assistant
          had not volunteered. I was suspicious of how quickly the first version came
          together, and treating that suspicion as a reason to ask was the most useful thing
          I did for this question.</p>
          <p>For parts a to c I asked for an audit against the handout's exact wording rather
          than against the assistant's memory of it. That is how the missing part 2b was
          found. I also required that every example be chosen by a rule that is printed next
          to it, so a reader can see the examples were not picked to flatter the models.</p>''',
        critique=f'''<p><b>What it got right.</b> Catching the hidden GLUE test labels was
          important: an accuracy score against labels of -1 still produces a plausible-looking
          number. The assistant made the loader refuse that split outright. It also checked
          the phrase corpus against the test and dev splits for leakage and removed the single
          overlapping row.</p>
          <p><b>What it overlooked.</b> The phrase-level corpus is described as 4.7 times more
          text, but {pt['rows_inside_frac']:.1%} of its rows are fragments of the 6,920
          training sentences. The model sees the same sentences again in pieces, not new
          sentences. The assistant did not say this until I questioned the training size.
          It also left part 2b out of the first submission build, and it never linked the data
          from the website until I asked.</p>
          <p><b>What it miscalculated.</b> The error analysis first said negation and contrast
          made up 38 to 45% of every model's errors, against about 20% of the test set. Once
          those figures were computed from the data rather than written into the text, both
          were wrong: the range is {min(shares):.1%} to {max(shares):.1%}, and the test set
          share is {test_share:.0%}. The conclusion (both categories are over-represented
          among errors) survives, but by less than claimed. For 2a, it could not find a
          published accuracy for the released GoogleNews file and said so, rather than
          attaching a number from a paper about a different model. I kept that.</p>'''),
    "q3": dict(
        asked="Show top-K ranked results for five test examples and compare with the "
              "pretrained model.",
        rounds=[
            ("The first report", "Build the report", "Deliverables 3 and 5 were both answered "
             "with one table of nearest words."),
            ("A TA's reply that sentiment analysis is expected when SST is used",
             "Should anything change?", "The assistant compared the two deliverables' wording "
             "and found that 3 asks for test examples, meaning sentences."),
            ("That finding", "Do everything you listed",
             "<code>src/testcases.py</code>: five test sentences chosen by rule, each with "
             "the classifier's ranked labels and the five nearest training sentences."),
            ("The finished section", "Why did we do sentiment analysis at all? Where is the "
             "model retrieving the information from?", "Explanations, and a clearer "
             "statement in the section of what retrieval shows."),
        ],
        strategy='''<p>I used the TA's answer as an outside check on the assistant's reading of
          the handout. Instead of asking it to add sentiment analysis, which it had already
          done, I asked whether anything should change, which made it re-read the handout next
          to the report. Later I asked basic questions about the purpose of the section and
          where the model gets its answers. Asking the obvious question is cheap, and when the
          answer takes a paragraph, the report probably needs one too.</p>''',
        critique='''<p><b>What it got right.</b> The ranked output per example has two parts:
          the classifier's labels with probabilities, and the nearest training sentences. The
          second part is what makes Example 5 interesting. All five models label it correctly
          and confidently, and retrieval shows why: its nearest training sentence is a
          near-copy with the same stacked negations. They are right because of word overlap,
          not because they understood the negation.</p>
          <p><b>What it overlooked.</b> It first merged deliverables 3 and 5 into one table of
          word neighbours, answering 5 twice and 3 not at all. Five examples is also a small
          sample: one example moves a model's score by 20 points, so the per-model tallies in
          this section describe these five sentences and nothing more. The full test set
          accuracy in Question 1 is the comparison that counts.</p>'''),
    "q4": dict(
        asked="Measure training cost, per-query latency and memory, and keep the "
              "measurements honest when the hardware changes.",
        rounds=[
            ("Phases 6 to 8 of the plan", "Do phases 6 to 8",
             "Latency per <code>most_similar</code> call, and a memory table checked against "
             "a prediction from matrix sizes. The first version measured the whole Python "
             "process; it was replaced with a per-model measurement in a separate process."),
            ("C on text8, laptop unusable", "Can I stop and resume? What if it is killed "
             "midway? Should this run on PACE ICE?", "Dense and sparse updates benchmarked, "
             "checkpoints added, and the job moved to an A100."),
        ],
        strategy='''<p>The plan I approved required device and memory choices to be measured
          rather than assumed. For Model C that paid off twice. At SST's vocabulary size the
          CPU beat the Apple GPU, because the matrices are too small for the GPU to pay off.
          On text8's vocabulary, ten times larger, the benchmark reversed, which is what sent
          the long run to a GPU.</p>
          <p>When the text8 run made my laptop unusable, I asked practical questions first:
          can I stop it and resume, and what happens if it dies halfway? Checkpointing came out
          of those questions and was verified by resuming a run and comparing weights before
          any long job depended on it.</p>''',
        critique='''<p><b>What it got right.</b> Latency grows with vocabulary size, as brute
          force search should, and the section says so with a measured ratio rather than a
          rule of thumb. It also kept training times from different machines out of the same
          comparison, since a fanless laptop and an A100 are not comparable.</p>
          <p><b>What it got wrong.</b> The first memory table compared the whole Python
          process against the size of the model's matrices and reported roughly 300 MB for a
          model of about 34 MB. The fix was a measurement per model, in its own process.</p>
          <p><b>What is still open.</b> Model G uses about a third of the memory its matrix
          size predicts. The likely cause is macOS memory compression, and the test is one run
          on a Linux node. That run has not been done, so the report calls it unexplained
          rather than explaining it. The laptop's training times also drift as it heats up,
          which the section notes.</p>'''),
    "q5": dict(
        asked="Explain the hyperparameters, separate the effect of each one, and compare "
              "top-1, top-5 and top-10 neighbours for five words.",
        rounds=[
            ("Phases 6 to 8 of the plan", "Do phases 6 to 8",
             "Neighbour tables for the five query words fixed in the plan: <code>plot</code>, <code>terrible</code>, <code>star</code>, "
             "<code>cinematography</code>, and <code>the</code> as a control."),
            ("The independent review", "Top-1, top-5 and top-10 must be explicit",
             "One table per query, with the three ranks in separate columns."),
            ("The live site", "Some entries have a gold tint that is never explained",
             "A legend: gold marks a G-ft neighbour that G does not have."),
            ("A list of possible extensions", "Do the hyperparameter sweep first",
             f"<code>src/sweep.py</code>: one factor at a time, three seeds each, "
             f"{len({json.dumps(x['runs']) for x in c.sweep['rows']})} distinct settings."),
        ],
        strategy='''<p>When I asked what in the list of possible extensions was worth doing, I
          chose the sweep first, because it tests the claims already in the report rather
          than adding new ones. I asked for three seeds per setting so that every comparison
          comes with a spread, and for one factor to change at a time so that each result can
          be attributed to its cause.</p>''',
        critique=f'''<p><b>What it got right.</b> The sweep separated architecture from loss.
          Changing the loss moves accuracy by almost nothing, while changing skip-gram to CBOW
          moves it by about {sgns - cbhs:.3f}. It also showed that no setting gives the SST
          models any word-similarity structure, which pins that failure on the amount of data
          rather than on a bad configuration.</p>
          <p><b>What it overlooked.</b> Before the sweep, every gap in the report came from
          a single run, and the assistant never mentioned the seed noise. With a spread of
          about half a point, several small differences it had described, including the
          effect of the number of negative samples, are not real results. It also rendered a
          gold highlight in the neighbour tables without saying what it meant, which I only
          noticed while reading the page as a reader would.</p>'''),
    "q6": dict(
        asked="Record the setup and running problems, deploy the report, and make the "
              "reproduction steps true.",
        rounds=[
            ("A new repository", "Create a git repo and push it to GitHub, like HW1",
             "A GitHub repository, later deployed from <code>public/</code> on Vercel."),
            ("The deploy", "This should be its own repo, like HW1, not part of my "
             "portfolio", "The assistant had placed it inside the portfolio repository; "
             "moved back to its own deployment."),
            ("Phase 0 of the plan", "Start on the implementation", "Python 3.11 instead of "
             "3.14, and scipy and numpy pinned after a misleading import error."),
            ("C on text8 making the laptop unusable", "Stop this run and do it on PACE", "A CUDA 12.6 environment, a job script "
             "that pins the GPU type and checks it before training."),
            ("A wrong sentence in the report", "Check for any more lines like this",
             "A README command that did nothing, a claim that four commands reproduce "
             "everything, and missing packages in <code>requirements.txt</code>, all fixed."),
            ("Last check before submitting", "Fix the submission blockers, mean latency, "
             "direct dataset link, and reproducibility commands", "Commands for G-ft, A2 and "
             "the example files, mean latency next to p50 and p95, a GLUE link, and a script "
             "that packages the ZIP."),
        ],
        strategy='''<p>For this question I relied on HW1 as a template: the same repository
          layout, the same deployment and the same cluster. Pointing the assistant at a
          working example was faster than describing what I wanted. Where it drifted from
          that example, as with the deployment, I stopped it and asked how it had done it
          before asking it to redo it.</p>''',
        critique='''<p><b>What it got right.</b> Each setup problem is recorded with the
          misleading symptom, not only the fix. The scipy problem shows up as an import error
          inside gensim, and the hidden-label split still produces an accuracy number. Those
          symptoms are what someone else would actually see.</p>
          <p><b>What it got wrong.</b> It put the project inside my portfolio repository when
          I had asked for a separate deployment like HW1. The README listed a fine-tuning
          command that ran nothing, the report said four commands reproduce every result when
          several results have no command at all, and <code>requirements.txt</code> left out
          PyTorch and pyarrow. None of these break the results, but each would have cost a
          grader time. G-ft, A2 and the 2b and 2c example files now have commands too, so
          the reproduction list covers every stored result.</p>'''),
    "q7": dict(
        asked="Cover the handout's query, analogy and visualization steps, and make each "
              "one readable.",
        rounds=[
            ("The finished report", "Did we do <code>most_similar</code>? Did we do "
             "analogies?", "Both were present; the assistant pointed to the sections."),
            ("The analogy section", "Shouldn't the analogies be about movies, since the "
             "models were trained on reviews?", "Four film-review analogies added, using "
             "only words that occur in SST."),
            ("The analogy tables", "Group by query rather than by model and highlight the "
             "top answer", "One table per query with the top answer marked."),
            ("The analogy results", "Why do A and B do so badly? Is that explained anywhere?",
             "An analysis of vocabulary coverage and word frequency added to the section."),
            ("A sentence about the analogy results", "This line is wrong. Check for others "
             "like it", "The sentence had gone stale when the text8 rows were added; 11 "
             "similar statements were found and fixed."),
            ("The visualization section", "I don't understand this section",
             "Rewritten in plain terms, text8 models added, and a check on the full vectors."),
        ],
        strategy='''<p>Most of my prompts here were reader complaints, not instructions: "this
          makes no sense to me", "I don't understand this section", "this line is incorrect".
          I read the report as someone seeing it for the first time and reported where I got
          lost. The film-review analogies came from asking whether the standard example was
          the right test for models trained only on movie reviews.</p>''',
        critique=f'''<p><b>What it got right.</b> The analogy section scores coverage as well
          as accuracy: gensim skips any question with an unknown word, so a model with a small
          vocabulary is graded on a smaller and different set of questions. The visualization
          rewrite added a measurement on the full vectors and found something the plots hide:
          the text8 models match GoogleNews on word similarity but separate positive from
          negative words worse than any other model.</p>
          <p><b>What it got wrong.</b> Numbers in the report are read from the data, but the
          sentences around them are not. When the text8 rows were added, "only G and G-ft
          solve the canonical analogy" became false, and so did ten other statements. The
          visualization section claimed the two plotted axes keep under 20% of the variance
          for every model; the actual range is {min(ev):.0%} to {max(ev):.0%}. Its first draft
          also explained how the words were chosen before explaining why the plots exist,
          which is why I could not follow it.</p>'''),
    }


def step2(c, qid, rec):
    rounds = c.table(["Round", "Context supplied", "What I asked", "Outcome"],
                     [[f"{i}", a, b, o] for i, (a, b, o) in enumerate(rec["rounds"], 1)],
                     ["m", "l", "l", "l"])
    return f'''<div class="steplabel" id="{qid}-s2"><span>Step 2</span>AI interaction record</div>
      <div class="airec">
        <div class="ai"><span class="k">1</span><div><b>AI tool</b>
          <p>Claude Code, <code>https://claude.ai/code</code>. See the
          <a href="#tools">shared record</a>.</p></div></div>
        <div class="ai"><span class="k">2</span><div><b>What I asked for</b>
          <p>{rec["asked"]}</p></div></div>
        <div class="ai"><span class="k">3</span><div><b>Workflow, round by round</b>
          <p>Each round's query is summarised, not quoted.</p>{rounds}</div></div>
        <div class="ai"><span class="k">4a</span><div><b>Prompting strategy</b>
          {rec["strategy"]}</div></div>
        <div class="ai"><span class="k">4b</span><div><b>Critique of the AI's response</b>
          {rec["critique"]}</div></div>
      </div>'''


REFS = [
    "T. Mikolov, K. Chen, G. Corrado and J. Dean. Efficient Estimation of Word Representations "
    "in Vector Space. ICLR Workshop, 2013. <a href='https://arxiv.org/abs/1301.3781'>arXiv:1301.3781</a>",
    "T. Mikolov, I. Sutskever, K. Chen, G. Corrado and J. Dean. Distributed Representations of "
    "Words and Phrases and their Compositionality. NIPS, 2013. "
    "<a href='https://arxiv.org/abs/1310.4546'>arXiv:1310.4546</a>",
    "F. Morin and Y. Bengio. Hierarchical Probabilistic Neural Network Language Model. "
    "AISTATS, 2005.",
    "R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng and C. Potts. Recursive "
    "Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP, 2013. "
    "<a href='https://nlp.stanford.edu/sentiment/'>nlp.stanford.edu/sentiment</a>",
    "A. Wang, A. Singh, J. Michael, F. Hill, O. Levy and S. Bowman. GLUE: A Multi-Task "
    "Benchmark and Analysis Platform for Natural Language Understanding. ICLR, 2019.",
    "R. Řehůřek and P. Sojka. Software Framework for Topic Modelling with Large Corpora. "
    "LREC Workshop on New Challenges for NLP Frameworks, 2010. "
    "<a href='https://radimrehurek.com/gensim/'>radimrehurek.com/gensim</a>",
    "L. Finkelstein, E. Gabrilovich, Y. Matias, E. Rivlin, Z. Solan, G. Wolfman and E. Ruppin. "
    "Placing Search in Context: The Concept Revisited. ACM TOIS 20(1), 2002. (WordSim-353)",
    "F. Hill, R. Reichart and A. Korhonen. SimLex-999: Evaluating Semantic Models with "
    "(Genuine) Similarity Estimation. Computational Linguistics 41(4), 2015.",
    "E. Bruni, N. K. Tran and M. Baroni. Multimodal Distributional Semantics. JAIR 49, 2014. "
    "(MEN-3k)",
    "L. van der Maaten and G. Hinton. Visualizing Data using t-SNE. JMLR 9, 2008.",
    "P. J. Rousseeuw. Silhouettes: a Graphical Aid to the Interpretation and Validation of "
    "Cluster Analysis. Journal of Computational and Applied Mathematics 20, 1987.",
    "D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. ICLR, 2015.",
    "F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.",
    "A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. "
    "NeurIPS, 2019.",
    "M. Mahoney. text8, the first 10<sup>8</sup> bytes of English Wikipedia, cleaned. "
    "<a href='http://mattmahoney.net/dc/textdata.html'>mattmahoney.net/dc/textdata.html</a>",
    "Google. word2vec, original C implementation and GoogleNews-vectors-negative300. "
    "<a href='https://code.google.com/archive/p/word2vec/'>code.google.com/archive/p/word2vec</a>",
    "SetFit/sst2 on the Hugging Face Hub, the sentence-level dev and test splits. "
    "<a href='https://huggingface.co/datasets/SetFit/sst2'>huggingface.co/datasets/SetFit/sst2</a>",
    "stanfordnlp/sst2 on the Hugging Face Hub, the GLUE copy whose train split supplies the "
    "67,349 labelled phrases. "
    "<a href='https://huggingface.co/datasets/stanfordnlp/sst2'>huggingface.co/datasets/stanfordnlp/sst2</a>",
    "Andras7/word2vec-pytorch, the PyTorch reference linked in the handout. "
    "<a href='https://github.com/Andras7/word2vec-pytorch'>github.com/Andras7/word2vec-pytorch</a>",
    "CS 6220 Fall 2026, Homework Assignment 2 (programming), and the course's template for "
    "AI-assisted homework.",
]


def assemble(buckets, titles, c):
    recs = records(c)
    parts = ["".join(buckets["head"]), contents(titles), front(c)]
    for qid, label, dl, ask, ids in QUESTIONS:
        parts.append(f'''<section id="{qid}" class="qband">
          <div class="eyebrow">{label} &middot; {dl}</div>
          <h2>{ask}</h2></section>
          <div class="steplabel"><span>Step 1</span>Final answer (AI generated)</div>''')
        parts += ["".join(buckets[i]) for i in ids]
        parts.append(step2(c, qid, recs[qid]))
    refs = "".join(f"<li>{x}</li>" for x in REFS)
    parts.append(f'''<section id="refs" class="qband">
      <div class="eyebrow">Step 3 &middot; outside the page limit</div><h2>References</h2></section>
      <ol class="refs">{refs}</ol>''')
    parts.append('''<section class="qband" id="appendix">
      <div class="eyebrow">Step 4 &middot; outside the page limit</div><h2>Appendices</h2>
      <p class="prose">Two analyses that go beyond the handout. Appendix A tests word
      similarity against human ratings, which shows what the sentiment score hides. Appendix
      B retrains A, B and C on a corpus 27 times larger.</p></section>''')
    for aid, label, ids in APPENDIX:
        parts.append(f'<div class="steplabel" id="{aid}"><span>{label}</span>{titles[ids[0]]}</div>')
        parts += ["".join(buckets[i]) for i in ids]
    parts.append("".join(buckets["foot"]))
    return "".join(parts)


def context(**kw):
    return SimpleNamespace(**kw)


SUB_CSS = """
  .tcover{display:grid;gap:6px;padding:28px 0 0;font-size:.9rem}
  .tcover>div{display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:baseline}
  .tcover span:first-child{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;
    text-transform:uppercase;color:var(--faint)}
  .sid{display:inline-block;min-width:140px;border-bottom:1px solid var(--line-2);height:1.2em}
  .toc-table td:first-child{width:52%}
  nav.toc a.tq{font-weight:600}
  .qband{margin-top:72px;padding:24px 26px;background:var(--surface);
    border:1px solid var(--line-2);border-left:4px solid var(--accent);border-radius:6px;gap:10px}
  .qband h2{font-size:1.35rem;line-height:1.3}
  .steplabel{display:flex;align-items:baseline;gap:12px;margin-top:44px;padding-bottom:10px;
    border-bottom:2px solid var(--ink);font-family:var(--serif);font-size:1.2rem;font-weight:600;
    scroll-margin-top:64px}
  .steplabel span{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
    text-transform:uppercase;color:var(--surface);background:var(--accent);padding:3px 8px;
    border-radius:3px;font-weight:500}
  .airec{display:flex;flex-direction:column;margin-top:8px}
  .ai{display:grid;grid-template-columns:46px 1fr;gap:16px;padding:20px 0;
    border-bottom:1px solid var(--line)}
  .ai .k{font-family:var(--mono);font-size:.8rem;color:var(--accent);padding-top:2px}
  .ai>div{display:flex;flex-direction:column;gap:10px;max-width:76ch}
  .ai>div>b{font-family:var(--sans);font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;
    color:var(--muted)}
  .ai p{color:var(--ink)}
  .ai .tablewrap{margin-top:4px}
  .ai table{font-size:.84rem}
  .ai td{vertical-align:top}
  .ai th:first-child,.ai td:first-child{width:52px}
  ol.refs{margin:18px 0 0;padding-left:22px;display:flex;flex-direction:column;gap:10px;
    font-size:.9rem;max-width:80ch}
  @media print{
    .qband{break-before:page;margin-top:0}
    #refs.qband,#appendix.qband{break-before:page}
    .steplabel{break-after:avoid}
    .ai{break-inside:auto}
    .ai .k{break-after:avoid}
    .toc-table{font-size:.85rem}
  }
"""
