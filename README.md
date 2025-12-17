# trm-llm

Codebase for TRM LLMs

Codebase structure so far:
- models/common contains two items: layers.py has some basic layers implemented (not optimized for computation) and trainer.py has the pytorch lightning pytorch trainer
- gpt.py is a gpt-2 vanilla model.
Then the levels are added based on how they're described in the paper
- gpt_level1.py reuses the same block instead of two
- gpt_level2.py adding step/time embedding
- UT is the universal transformer

Then the changes from UT toward TRM are also implemented in 2 sub-models and resulting in a TRM.
- ut_level1. Decoupling reasoning from solution.
- ut_level2. Some other stuff I Can't remember now.
- trm

Examples to launch the code:
```
python train.py \
  --model ut\
  --dataset addition_char \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
  --algo-train-len 20 \
  --dropout 0.1 \
  --gpu 1 \
  --algo-eval-extrap-len 40


python train.py \
  --model trm\
  --dataset addition_char \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
  --algo-train-len 20 \
  --dropout 0.1 \
  --gpu 0 \
  --algo-eval-extrap-len 40
  ```

  You might benefit from writing `pip install -e .`
