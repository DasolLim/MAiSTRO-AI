import Link from "next/link";
import { StaffDivider } from "@/components/StaffDivider";

const ARCHITECTURE = [
  {
    step: "1",
    title: "Bidirectional LSTM layer",
    description:
      "Reads the note sequence both forward (start to end) and backward (end to start), then combines both readings into a single, better-informed representation.",
  },
  {
    step: "2",
    title: "Dropout layer",
    description: "Randomly ignores a portion of the network during training, used throughout the model to keep it from overfitting.",
  },
  {
    step: "3",
    title: "Attention layer",
    description:
      "Weighs which parts of the sequence actually matter to the next note, the way a person skims a sentence for the load-bearing words instead of reading every one equally.",
  },
  {
    step: "4",
    title: "LSTM layer",
    description: "A second pass that folds the attention-weighted context back into the sequence.",
  },
  {
    step: "5",
    title: "Dropout layer",
    description: "A second guard against overfitting before the final decision is made.",
  },
  {
    step: "6",
    title: "Dense layer",
    description: "Organizes every note and chord the model has learned into a single set of candidates.",
  },
  {
    step: "7",
    title: "Softmax activation",
    description: "Turns that set of candidates into probabilities, so the model can pick the most likely next note or chord.",
  },
] as const;

const RESULTS = [
  {
    epoch: "Epoch 1",
    loss: null,
    src: "/media/audio/epoch-1.mp3",
    caption:
      "That definitely sounded like the first time someone tried playing the piano. Clearly, it hasn't learned much from the dataset yet.",
  },
  {
    epoch: "Epoch 10",
    loss: null,
    src: "/media/audio/epoch-10.mp3",
    caption:
      "The model has picked up some more complex chords and started developing patterns, but still struggles to vary its rhythm.",
  },
  {
    epoch: "Epoch 20",
    loss: null,
    src: "/media/audio/epoch-20.mp3",
    caption:
      "Noticeably more melodic coherence here, and we're finally seeing some variation in note lengths, although still limited.",
  },
  {
    epoch: "Epoch 50",
    loss: "0.1534",
    src: "/media/audio/epoch-50.mp3",
    caption:
      "Arguably the best-sounding output we generated, though at a loss this low, there's a good chance some of it is memorized rather than composed.",
  },
] as const;

export default function Home() {
  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        Program notes
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground sm:text-5xl">
        How MAiSTRO came to be
      </h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        An LSTM composer trained on Mozart, Beethoven, and Chopin, built by a team of eight at
        Western Cyber Society. The story so far, in the order we lived it.
      </p>

      {/* Motivation */}
      <section className="mt-20">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Origins
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">Motivation</h2>
        <div className="mt-5 max-w-[70ch] space-y-4 text-muted-foreground">
          <p>
            Outside of engineering, music has always been a huge part of my life, listening to
            it, picking apart how a track is built, appreciating the work that goes into
            composing something that actually moves people. I&apos;ve always been fascinated by how
            music artists compose, layering melody, rhythm, and emotion into something that
            connects with listeners in a way that&apos;s hard to put into words. That fascination made
            me wonder how I could bring my engineering skills into that world instead of just
            enjoying it from the outside.
          </p>
          <p>
            At the same time, AI was exploding everywhere: essays written in seconds, images that
            made you question what was even real anymore. It felt like AI was closing in on doing
            anything a human could do. But one space seemed strangely untouched: music
            composition. That gap felt like the perfect intersection of two things I wanted to
            explore, could AI genuinely create music the way a person could, and could I build
            something that gave engineers a real way to support musicians instead of sidelining
            them?
          </p>
        </div>
      </section>

      <StaffDivider className="mt-16" />

      {/* Our Plan */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          The plan
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">Our plan</h2>
        <p className="mt-5 max-w-[70ch] text-muted-foreground">
          We decided to start simple: build a machine learning model that could learn from
          classical piano music and generate its own compositions. This meant finding a suitable
          dataset in a format the model could understand, selecting the right model
          architecture, running inference to generate new sequences, and finally converting
          those outputs back into audible music.
        </p>
        {/* eslint-disable-next-line @next/next/no-img-element -- static content diagram, size unknown until the asset is supplied */}
        <img
          src="/media/diagrams/pipeline.png"
          alt="Pipeline: existing music (.mp3) is converted to MIDI data, fed into the model, which generates new MIDI data, converted back into audible music (.mp3)."
          className="mt-8 w-full border border-border"
        />
      </section>

      <StaffDivider className="mt-16" />

      {/* Data Collection & Preprocessing */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          The dataset
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">Data collection &amp; preprocessing</h2>
        <div className="mt-5 max-w-[70ch] space-y-4 text-muted-foreground">
          <p>
            For the dataset, we gathered a collection of piano pieces by Mozart, Beethoven, and
            Chopin since they are easily recognizable, which made checking for overfitting much
            more straightforward. Initially, we thought we would need to convert audio recordings
            of these compositions into a format the model could understand. However, because
            these pieces are so well-known, computer-readable versions already exist online.
            These files, known as MIDI files, essentially represent digital sheet music, making
            them perfect for feeding into our model.
          </p>
        </div>

        {/* eslint-disable-next-line @next/next/no-img-element -- static content diagram, size unknown until the asset is supplied */}
        <img
          src="/media/diagrams/data-collection.png"
          alt="Sheet music alongside its MIDI representation, shown as a piano roll and as raw note/duration text like '64 1.0_74 0.25_74 0.25'."
          className="mt-8 w-full border border-border"
        />
      </section>

      <StaffDivider className="mt-16" />

      {/* The Model */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          The model
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">The model</h2>

        <div className="mt-8 max-w-[70ch] space-y-4 text-muted-foreground">
          <p>
            When planning this project, Henry and I had a few contenders for models we could use
            to accomplish the task. Autoencoders can be used by learning compressed
            representations of musical patterns and then decoding them to create new
            compositions. Generative Adversarial Networks (GANs), consisting of two models in
            competition, one generating music and the other evaluating its realism, also
            presented a compelling option. Lastly, Recurrent Neural Networks (RNNs) work by
            learning patterns in sequences of musical notes and predicting the next note based on
            previously played ones. After doing research, we felt that RNNs most closely mimicked
            the way humans process and compose music, leading us to choose a specific variant
            known as the Long Short-Term Memory (LSTM) model for our implementation.
          </p>
          <p>
            You would undoubtedly ask, &ldquo;Why not just use a regular RNN?&rdquo; While RNNs and LSTMs
            essentially aim to accomplish the same task, standard RNNs have a much smaller memory
            capacity compared to LSTMs. For example, if I asked an RNN, &ldquo;The sky is…&rdquo;, it would
            (hopefully) respond with &ldquo;blue.&rdquo; But if I asked it, &ldquo;Explain why the sky is blue,&rdquo; by
            the second sentence it would start rambling about how the sky is actually just a
            reflection of the ocean and that&apos;s why deserts don&apos;t exist at night. This is because
            of a common mathematical issue called the vanishing gradient problem, where the
            gradient of the loss function decays exponentially over time. This is fine if the
            model is outputting one-word answers, but worse if the answers are meant to be longer
            since the model will forget more information faster and faster as the length of the
            answer increases.
          </p>
          <p>
            LSTMs help solve this problem by allowing a larger memory of past inputs. An
            additional enhancement to these models is the attention mechanism. While RNNs and
            LSTMs use previous inputs to influence future outputs, they lack the ability to label
            what parts of previous inputs are important to a specific output. This is important
            in music generation, where the model must understand structural elements like verses
            and choruses to repeat or develop them appropriately. The attention mechanism allows
            the model to focus on these significant sections, improving its ability to generate
            musically structured compositions. Here&apos;s an example that helped me understand what
            the model was doing:
          </p>
          <p className="font-display text-lg italic text-foreground">
            Unless you&apos;ve seen this joke before, you probably fell for it (like I did… more times
            than I&apos;d like to admit).
          </p>
          <p>
            As humans, we don&apos;t read word-by-word. Instead, we read the most important parts and
            assume where the sentence is going from there. This is what attention is doing within
            our model: guiding the LSTM in what it needs to remember or forget as time goes on.
          </p>
          <p>Our final model had the following architecture:</p>
        </div>

        <div className="mt-8">
          <StaffDivider />
          {ARCHITECTURE.map((layer) => (
            <div key={layer.step}>
              <div className="flex items-baseline gap-6 py-6">
                <span className="w-8 shrink-0 font-display text-xl italic text-muted-foreground">
                  {layer.step}
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="font-display text-xl text-foreground">{layer.title}</h3>
                  <p className="mt-1.5 max-w-[60ch] text-sm text-muted-foreground">
                    {layer.description}
                  </p>
                </div>
              </div>
              <StaffDivider />
            </div>
          ))}
        </div>
      </section>

      <StaffDivider className="mt-16" />

      {/* Results */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Results
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">
          Alright, let&apos;s hear how it sounds
        </h2>
        <p className="mt-5 max-w-[70ch] text-muted-foreground">
          I&apos;m done with the theory. Here&apos;s the model at four points along the same training
          run.
        </p>

        <div className="mt-10 space-y-12">
          {RESULTS.map((result) => (
            <div key={result.epoch}>
              <div className="flex items-baseline justify-between gap-4">
                <h3 className="font-display text-2xl italic text-foreground">{result.epoch}</h3>
                {result.loss && (
                  <span className="text-xs tabular-nums text-muted-foreground uppercase">
                    loss {result.loss}
                  </span>
                )}
              </div>
              <p className="mt-2 max-w-[65ch] text-sm text-muted-foreground">{result.caption}</p>
              <audio controls src={result.src} className="mt-4 w-full">
                Your browser does not support the audio element.
              </audio>
            </div>
          ))}
        </div>

        <p className="mt-12 max-w-[70ch] text-muted-foreground">
          By the way, the model only outputs MIDI files during generation, to turn one into
          something audible, we manually applied a piano soundfont, a process the{" "}
          <Link href="/generate" className="text-brass underline underline-offset-2">
            Generate
          </Link>{" "}
          and{" "}
          <Link href="/library" className="text-brass underline underline-offset-2">
            Listen
          </Link>{" "}
          movements above now do automatically. With the remaining Google Colab credits we had,
          we trained the model for a final 50 epochs, that&apos;s the last clip above. This is
          something I would genuinely listen to if I was trying to zone in during a study
          session, at a cafe, or any other situation where background music would fit in.
        </p>
      </section>

      <StaffDivider className="mt-16" />

      {/* Closing CTA */}
      <section className="mt-16 mb-4">
        <p className="max-w-[65ch] text-muted-foreground">
          That&apos;s the story so far. The tool itself is above, start with{" "}
          <Link href="/dataset" className="text-brass underline underline-offset-2">
            Prepare
          </Link>{" "}
          if you&apos;re bringing your own dataset, or jump straight to{" "}
          <Link href="/generate" className="text-brass underline underline-offset-2">
            Generate
          </Link>{" "}
          if a model is already trained.
        </p>
      </section>
    </div>
  );
}
