# Tmux Training Guide

## Quick Start

Start all TL-1 experiments in tmux:
```bash
bash scripts/run_in_tmux.sh
```

Start a single experiment:
```bash
bash scripts/run_in_tmux.sh tl1_lr_1e4
```

## Essential Tmux Commands

### Session Management
```bash
# Attach to training session
tmux attach -t tl1_baseline

# Detach from session (training continues)
# Press: Ctrl+b, then d

# List all sessions
tmux ls

# Kill session (stops training)
tmux kill-session -t tl1_baseline
```

### Inside Tmux Session
```
Ctrl+b d        Detach from session
Ctrl+b [        Scroll mode (use arrows, q to exit)
Ctrl+b c        Create new window
Ctrl+b n        Next window
Ctrl+b p        Previous window
```

## Monitor Training

### View live logs
```bash
# Find latest log file
ls -lth logs/

# Monitor in real-time
tail -f logs/tl1_all_*.log
```

### Check progress from outside tmux
```bash
# Quick peek at session
tmux capture-pane -t tl1_baseline -p | tail -20

# Or attach to see full output
tmux attach -t tl1_baseline
```

## Training Status

### Check if training is running
```bash
# List tmux sessions
tmux ls

# Check GPU usage
nvidia-smi

# Check if process is active
ps aux | grep train_tl1
```

### If training crashes
```bash
# Check logs
tail -100 logs/tl1_all_*.log

# Check results
ls -lah runs/tl1_baseline/*/weights/
```

## Best Practices

1. **Before disconnecting**: Make sure tmux session is running
   ```bash
   tmux ls
   ```

2. **Monitor occasionally**: Check logs or attach to session
   ```bash
   tmux attach -t tl1_baseline
   # Then detach: Ctrl+b d
   ```

3. **Long training runs**: Use tmux for all experiments > 1 hour

4. **After completion**: Session will remain; attach to see results
   ```bash
   tmux attach -t tl1_baseline
   ```

## Troubleshooting

### Session already exists
```bash
# Option 1: Attach to existing
tmux attach -t tl1_baseline

# Option 2: Kill old session first
tmux kill-session -t tl1_baseline
bash scripts/run_in_tmux.sh
```

### Can't find logs
```bash
# Logs are in:
ls -lth logs/

# Most recent:
tail -f logs/tl1_all_*.log
```

### Training seems stuck
```bash
# Check GPU usage
nvidia-smi

# Attach and check
tmux attach -t tl1_baseline
```
