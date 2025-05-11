# Hunting for Induction Heads in Amazon's Chronos Model

This is the codebase for my blog post on finding induction heads in [Amazon's Chronos](https://github.com/amazon-science/chronos-forecasting) probabilistic time series forecasting models.

## Usage
To get started, simply clone the repo and install the required packages
```bash
git clone https://github.com/hasithv/chronos_induction
cd chronos_induction
pip install -r requirements.txt
```
## Files
- `attention_plot.ipynb`: Visualizes attention for the `t5-small` model on an airline passengers dataset
- `attn_lens.py`: File to help store attention scores during inference
- `rrt_attention_mosiac.ipynb`: Creates the induction mosaic on repeated random tokens
- `rrt_utils.py`: Helper file to create repeated random token sequences
