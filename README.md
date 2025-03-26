# CS 224N Default Final Project: Building and Optimizing GPT-2

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course
website for detailed instructions and an overview of the codebase.

# Optimizations

We implemented LoRA, SMART and SOAP (three difficult optimization/regularlization techniques) to increase model efficiency while maintaining accuracy. For full details of our project, see the full paper.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).
